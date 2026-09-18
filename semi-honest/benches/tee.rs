//! Bench: Ladon TEE Asset Key Derivation (Algorithm 3 of the paper).
//!
//! Times the TEE-local work that runs AFTER the KBS committee has produced
//! its per-party (mu'*m)_j shares. That is:
//!
//!     mu_m_shares (input, from committee)
//!         |
//!         v
//!     receiver_reconstruct  (Step 6 + ell-fold majority decoding -> m_bytes)
//!         |
//!         v
//!     FO finalization:
//!         (K', r') = G(m' || H(ek))
//!         c'        = K-PKE.Encrypt(ek, m', r')   <-- dominant cost
//!         return K' if c == c', else J(z || c)
//!         |
//!         v
//!     asset key K_A
//!
//! Threshold convention (matches the demo):
//!   t = THRESHOLD = active-committee size = min-to-decrypt.
//!     - The first t parties (ids 1..=t) form the active committee.
//!     - Dealer creates t additive double sharings, one per committee member.
//!     - Dealer also creates n Shamir key shares (any t reconstruct sk),
//!       providing offline robustness. Here n = 2t + 1.
//!
//! Build & run:
//!     cargo run --release --bin bench_tee

#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use std::hint::black_box;
use std::time::{Duration, Instant};

use bitvec::view::BitView;
use rand::{rngs::StdRng, RngCore, SeedableRng};

use Ladon::crypt;
use Ladon::dealer::Dealer;
use Ladon::kpke;
use Ladon::mlkem::{self, MlKemCyphertext, MlKemEncapsulationKey};
use Ladon::params::*;
use Ladon::party::Party;
use Ladon::ring::{Compressed, Ring};
use Ladon::serialize::{BitOrder, MlKemDeserialize, MlKemSerialize};
use Ladon::threshold::{assemble_parties, receiver_reconstruct, threshold_decrypt};

// ===========================================================================
// ====== BENCH KNOBS — edit these ==========================================
// ===========================================================================
const THRESHOLDS: &[usize] = &[4, 8, 16, 32];
const P_PLAINTEXT: u64 = 2;

// Parallel double sharings per security level (Section 5.1).
const ELL_LADON128: usize = 31;
const ELL_LADON256: usize = 31;

const ITERATIONS: usize = 5000;
const WARMUP: usize = 100;
// ===========================================================================

fn main() {
    println!("==================================================================");
    println!("  Ladon Bench: TEE Asset Key Derivation");
    println!("==================================================================");
    println!();
    println!("  What this bench measures:");
    println!("    The TEE-local work performed inside the enclave AFTER the KBS");
    println!("    committee has finished its distributed protocol. Each timed");
    println!("    iteration runs:");
    println!("      1. receiver_reconstruct  - sum the per-party (mu'*m) shares");
    println!("                                  across the committee, decode each");
    println!("                                  of the ell candidates, and take");
    println!("                                  the coefficient-wise majority to");
    println!("                                  recover the 32-byte seed m'.");
    println!("      2. m' -> Compressed<1, Ring>  for re-encryption.");
    println!("      3. (K', r') = G(m' || H(ek))");
    println!("      4. c' = K-PKE.Encrypt(ek, m', r')          <-- dominant cost");
    println!("      5. if c == c': return K'  else: return J(z || c)");
    println!();
    println!("  EXCLUDED (these belong in other benches):");
    println!("      - The committee's distributed protocol (bench_committee_local).");
    println!("      - The asset owner's encaps             (bench_encapsulation).");
    println!();
    println!("    mu_m_shares is produced ONCE in setup by running the full");
    println!("    committee protocol; the same buffer is reused for every timed");
    println!("    iteration.");
    println!();
    println!("  Convention: t = active-committee size = min-to-decrypt. n = 2t + 1.");
    println!();
    run::<Ladon128>("Ladon128", ELL_LADON128);
    println!();
    run::<Ladon256>("Ladon256", ELL_LADON256);
    println!("==================================================================");
}

fn random_32() -> [u8; 32] {
    let mut rng = StdRng::from_entropy();
    let mut out = [0u8; 32];
    rng.fill_bytes(&mut out);
    out
}

fn run<PARAMS: MlKemParams>(label: &str, ell: usize)
where
    [(); 960 * PARAMS::K + 32]:,
    [(); 1920 * PARAMS::K + 96]:,
    [(); PARAMS::K]:,
    [(); PARAMS::ETA_1]:,
    [(); PARAMS::ETA_2]:,
    [(); 64 * PARAMS::ETA_1]:,
    [(); 64 * PARAMS::ETA_2]:,
    [(); 32 * (PARAMS::D_U * PARAMS::K + PARAMS::D_V)]:,
{
    println!("------------------------------------------------------------------");
    println!("  {label} parameter set");
    println!("------------------------------------------------------------------");
    println!("    K     (module rank)         : {}", PARAMS::K);
    println!("    eta_1 (key CBD width)       : {}", PARAMS::ETA_1);
    println!("    eta_2 (noise CBD width)     : {}", PARAMS::ETA_2);
    println!("    d_u   (u-compression bits)  : {}", PARAMS::D_U);
    println!("    d_v   (v-compression bits)  : {}", PARAMS::D_V);
    println!("    q     (ring modulus)        : {Q}");
    println!("    N     (ring degree)         : 256");
    println!("    p     (plaintext modulus)   : {P_PLAINTEXT}");
    println!("    ell   (parallel sharings)   : {ell}");
    println!("    warmup / timed iterations   : {WARMUP} / {ITERATIONS}");
    println!();
    println!(
        "  {:>5}  {:>5}    {:>14}    {:>14}    {:>12}",
        "t", "n", "total", "avg/op", "ops/sec"
    );
    println!(
        "  {:>5}  {:>5}    {:>14}    {:>14}    {:>12}",
        "---", "---", "--------------", "--------------", "------------"
    );

    for &t in THRESHOLDS {
        let n = 2 * t + 1;
        bench_one::<PARAMS>(t, n, ell);
    }
}

fn bench_one<PARAMS: MlKemParams>(t: usize, n: usize, ell: usize)
where
    [(); 960 * PARAMS::K + 32]:,
    [(); 1920 * PARAMS::K + 96]:,
    [(); PARAMS::K]:,
    [(); PARAMS::ETA_1]:,
    [(); PARAMS::ETA_2]:,
    [(); 64 * PARAMS::ETA_1]:,
    [(); 64 * PARAMS::ETA_2]:,
    [(); 32 * (PARAMS::D_U * PARAMS::K + PARAMS::D_V)]:,
{
    // ===== SETUP (untimed) ================================================
    // t = active-committee size = min-to-decrypt. Dealer creates t additive
    // double shares and n Shamir key shares (polynomial degree t-1, computed
    // internally by Dealer::new).
    let dealer = Dealer::new(t, n, P_PLAINTEXT);
    let ks = dealer.generate_keypair::<PARAMS>();
    let dbl = dealer.generate_double_sharing(ell);

    let ek = ks.ek.clone();
    let hash = crypt::h(&ek.clone().serialize().into_vec());
    let z = random_32();

    let (_key_b, c) = mlkem::encaps::<PARAMS>(ek.clone());

    // Active committee is implicit (first t parties). assemble_parties pairs
    // the first t Shamir shares with the t additive double shares.
    let parties: Vec<Party<{ PARAMS::K }>> =
        assemble_parties::<{ PARAMS::K }>(&ks.sk_shares, &dbl, dealer.thr);

    let mu_m_shares = threshold_decrypt(&parties, c.clone(), &dealer.thr);

    // ===== WARMUP =========================================================
    for _ in 0..WARMUP {
        let _ = tee_pipeline::<PARAMS>(
            &mu_m_shares,
            &dealer.thr,
            ek.clone(),
            hash,
            z,
            c.clone(),
        );
    }

    // ===== TIMED LOOP =====================================================
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let k = tee_pipeline::<PARAMS>(
            &mu_m_shares,
            &dealer.thr,
            ek.clone(),
            hash,
            z,
            c.clone(),
        );
        black_box(k);
    }
    let total = start.elapsed();
    let avg = total / ITERATIONS as u32;
    let ops_per_sec = ITERATIONS as f64 / total.as_secs_f64();

    println!(
        "  {:>5}  {:>5}    {:>14}    {:>14}    {:>12.2}",
        t,
        n,
        format_duration(total),
        format_duration(avg),
        ops_per_sec
    );
}

#[inline(never)]
fn tee_pipeline<PARAMS: MlKemParams>(
    mu_m_shares: &[Vec<[u64; 256]>],
    thr: &Ladon::dealer::ThrParams,
    ek: MlKemEncapsulationKey<{ PARAMS::K }>,
    hash: [u8; 32],
    z: [u8; 32],
    c: MlKemCyphertext<{ PARAMS::K }, { PARAMS::D_U }, { PARAMS::D_V }>,
) -> [u8; 32]
where
    [(); 960 * PARAMS::K + 32]:,
    [(); PARAMS::K]:,
    [(); PARAMS::ETA_1]:,
    [(); PARAMS::ETA_2]:,
    [(); 64 * PARAMS::ETA_1]:,
    [(); 64 * PARAMS::ETA_2]:,
    [(); 32 * (PARAMS::D_U * PARAMS::K + PARAMS::D_V)]:,
{
    let m_bytes: [u8; 32] = receiver_reconstruct(mu_m_shares, thr);

    let m: Compressed<1, Ring> =
        Compressed::<1, Ring>::deserialize(&m_bytes.view_bits::<BitOrder>().to_bitvec());

    let mut combined = [0u8; 64];
    combined[..32].copy_from_slice(&m_bytes);
    combined[32..].copy_from_slice(&hash);
    let (key, rand) = crypt::g::<64>(&combined);

    let c_prime = kpke::encrypt::<PARAMS>(ek, m, rand);

    if c.0 == c_prime.0 && c.1 == c_prime.1 {
        key
    } else {
        crypt::j([&z, c.serialize().as_raw_slice()].concat())
    }
}

fn format_duration(d: Duration) -> String {
    let nanos = d.as_nanos();
    if nanos >= 1_000_000_000 {
        format!("{:.6} s", nanos as f64 / 1_000_000_000.0)
    } else if nanos >= 1_000_000 {
        format!("{:.3} ms", nanos as f64 / 1_000_000.0)
    } else if nanos >= 1_000 {
        format!("{:.3} us", nanos as f64 / 1_000.0)
    } else {
        format!("{nanos} ns")
    }
}