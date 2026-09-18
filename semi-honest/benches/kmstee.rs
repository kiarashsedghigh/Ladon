//! Bench: Ladon Combined Committee + TEE Cost (per decryption request).
//!
//! This bench does NOT introduce any new measurement logic or parameters. It
//! simply RE-RUNS, unchanged, the two existing benches:
//!
//!   * bench_committee_local  (benches/committee_local.rs)
//!       -> the per-party local compute a KBS committee member spends on one
//!          decryption request (avg / party-op). This cost is essentially
//!          t-independent, so it is measured once per security level using the
//!          same fixed convention as the source bench (SETUP_T / SETUP_N).
//!
//!   * bench_tee              (benches/tee.rs)
//!       -> the TEE-local asset-key derivation performed after the committee
//!          has produced its shares (avg / op), swept over the same thresholds.
//!
//! For each security level it prints one table. Per threshold row:
//!
//!     committee/party   =  avg / party-op   (from committee_local.rs)
//!     tee/op            =  avg / op         (from tee.rs)
//!     sum               =  committee/party  +  tee/op
//!
//! The two source constants, setups, warmup counts, iteration counts and timed
//! pipelines are copied verbatim below so the produced numbers match running
//! each source bench on its own; only the presentation (adding the two
//! columns) is new.
//!
//! Build & run:
//!     cargo bench --bench kmstee
//!     cargo run --release --bin bench_kmstee

#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use std::hint::black_box;
use std::time::{Duration, Instant};

use bitvec::view::BitView;
use rand::{rngs::StdRng, RngCore, SeedableRng};

use Ladon::crypt;
use Ladon::dealer::{Dealer, ThrParams};
use Ladon::kpke;
use Ladon::mlkem::{self, MlKemCyphertext, MlKemEncapsulationKey};
use Ladon::params::*;
use Ladon::party::Party;
use Ladon::ring::{Compressed, Ring};
use Ladon::serialize::{BitOrder, MlKemDeserialize, MlKemSerialize};
use Ladon::threshold::{assemble_parties, open_v, receiver_reconstruct, threshold_decrypt};

// ===========================================================================
// ====== BENCH KNOBS — copied verbatim from the two source benches. ========
// ======  Do NOT edit: these must match committee_local.rs and tee.rs. =====
// ===========================================================================

// ell — parallel double sharings per security level (Section 5.1).
// (Identical value in both source benches.)
const ELL_LADON128: usize = 31;
const ELL_LADON256: usize = 31;

const P_PLAINTEXT: u64 = 2;

// ---- from tee.rs ----------------------------------------------------------
const THRESHOLDS: &[usize] = &[4, 8, 16, 32];
const TEE_ITERATIONS: usize = 5000;
const TEE_WARMUP: usize = 100;

// ---- from committee_local.rs ---------------------------------------------
// Per-party cost is essentially t-independent, so the source bench fixes t.
const SETUP_T: usize = 4;
const SETUP_N: usize = 9; // n >= t; satisfies SETUP_T <= SETUP_N
const COMMITTEE_ITERATIONS: usize = 5000;
const COMMITTEE_WARMUP: usize = 500;
// ===========================================================================

fn main() {
    println!("==================================================================");
    println!("  Ladon Bench: Combined Committee + TEE Cost");
    println!("==================================================================");
    println!();
    println!("  This bench re-runs, unchanged, two existing benches and prints");
    println!("  their per-op timings side by side plus their sum:");
    println!();
    println!("    committee/party : bench_committee_local avg / party-op");
    println!("                      (KBS member local compute for one request;");
    println!("                       essentially t-independent, measured once per");
    println!("                       security level at t = {SETUP_T}, n = {SETUP_N}).");
    println!("    tee/op          : bench_tee avg / op");
    println!("                      (TEE asset-key derivation, swept over t).");
    println!("    sum             : committee/party + tee/op.");
    println!();
    println!("  No source parameters or timed logic were modified; only the two");
    println!("  numbers are combined into one table.");
    println!();
    println!("  Convention: t = active-committee size = min-to-decrypt. n = 2t + 1");
    println!("  for the TEE sweep (as in tee.rs).");
    println!();
    run::<Ladon128>("Ladon128", ELL_LADON128);
    println!();
    run::<Ladon256>("Ladon256", ELL_LADON256);
    println!("==================================================================");
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
    println!(
        "    committee t / n             : {SETUP_T} / {SETUP_N}    (t-independent, measured once)"
    );
    println!(
        "    committee warmup / iters    : {COMMITTEE_WARMUP} / {COMMITTEE_ITERATIONS}"
    );
    println!("    tee warmup / iters          : {TEE_WARMUP} / {TEE_ITERATIONS}");
    println!();

    // ---- Committee: measured once (per-party cost is t-independent) -------
    let committee_avg = measure_committee::<PARAMS>(ell);

    println!(
        "  {:>5}  {:>5}    {:>16}    {:>16}    {:>16}",
        "t", "n", "committee/party", "tee/op", "sum"
    );
    println!(
        "  {:>5}  {:>5}    {:>16}    {:>16}    {:>16}",
        "---", "---", "----------------", "----------------", "----------------"
    );

    for &t in THRESHOLDS {
        let n = 2 * t + 1;
        let tee_avg = measure_tee::<PARAMS>(t, n, ell);
        let sum = committee_avg + tee_avg;
        println!(
            "  {:>5}  {:>5}    {:>16}    {:>16}    {:>16}",
            t,
            n,
            format_duration(committee_avg),
            format_duration(tee_avg),
            format_duration(sum)
        );
    }
}

// ===========================================================================
// ====== Committee measurement — copied verbatim from committee_local.rs ====
// ====== (returns avg / party-op instead of printing) =======================
// ===========================================================================
fn measure_committee<PARAMS: MlKemParams>(ell: usize) -> Duration
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
    let dealer = Dealer::new(SETUP_T, SETUP_N, P_PLAINTEXT);
    let ks = dealer.generate_keypair::<PARAMS>();
    let dbl = dealer.generate_double_sharing(ell);

    let (_key_b, c) = mlkem::encaps::<PARAMS>(ks.ek.clone());

    let parties: Vec<Party<{ PARAMS::K }>> =
        assemble_parties::<{ PARAMS::K }>(&ks.sk_shares, &dbl, dealer.thr);
    let active_ids: Vec<u32> = parties.iter().map(|p| p.id).collect();
    let v_holder_id = *active_ids.iter().min().expect("need >=1 party");

    // Precompute the opened e_tilde (simulating the network round once).
    let step1_all: Vec<_> = parties
        .iter()
        .map(|p| {
            let is_holder = p.id == v_holder_id;
            p.partial_decrypt_compressed::<{ PARAMS::D_U }, { PARAMS::D_V }>(
                c.clone(),
                &active_ids,
                is_holder,
            )
        })
        .collect();
    let masked_all: Vec<Vec<[u64; 256]>> = parties
        .iter()
        .zip(step1_all.iter())
        .map(|(p, out)| p.mask(&out.e_prime))
        .collect();
    let e_tilde_opened: Vec<[u64; 256]> = open_v(&masked_all, &dealer.thr);

    // Sanity: end-to-end protocol still works for this ell.
    {
        let mu_m_shares = threshold_decrypt(&parties, c.clone(), &dealer.thr);
        assert_eq!(mu_m_shares[0].len(), ell);
        black_box(mu_m_shares);
    }

    let target = &parties[1]; // non-v-holder (common case)
    let is_holder = target.id == v_holder_id;
    debug_assert!(!is_holder);

    // ===== WARMUP =========================================================
    for _ in 0..COMMITTEE_WARMUP {
        let k = committee_pipeline::<PARAMS>(
            target,
            &c,
            &active_ids,
            &dealer.thr,
            &e_tilde_opened,
            is_holder,
        );
        black_box(k);
    }

    // ===== TIMED LOOP =====================================================
    let start = Instant::now();
    for _ in 0..COMMITTEE_ITERATIONS {
        let k = committee_pipeline::<PARAMS>(
            target,
            &c,
            &active_ids,
            &dealer.thr,
            &e_tilde_opened,
            is_holder,
        );
        black_box(k);
    }
    let total = start.elapsed();
    total / COMMITTEE_ITERATIONS as u32
}

#[inline(never)]
fn committee_pipeline<PARAMS: MlKemParams>(
    party: &Party<{ PARAMS::K }>,
    c: &Ladon::mlkem::MlKemCyphertext<{ PARAMS::K }, { PARAMS::D_U }, { PARAMS::D_V }>,
    active_ids: &[u32],
    thr: &ThrParams,
    e_tilde_opened: &[[u64; 256]],
    is_holder: bool,
) -> Vec<[u64; 256]>
where
    [(); PARAMS::K]:,
    [(); 32 * (PARAMS::D_U * PARAMS::K + PARAMS::D_V)]:,
{
    let _ = thr;

    let step1 = party.partial_decrypt_compressed::<{ PARAMS::D_U }, { PARAMS::D_V }>(
        c.clone(),
        active_ids,
        is_holder,
    );

    let masked = party.mask(&step1.e_prime);
    black_box(&masked);

    party.finalize(&step1.w_prime, e_tilde_opened, is_holder)
}

// ===========================================================================
// ====== TEE measurement — copied verbatim from tee.rs ======================
// ====== (returns avg / op instead of printing) =============================
// ===========================================================================
fn measure_tee<PARAMS: MlKemParams>(t: usize, n: usize, ell: usize) -> Duration
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
    let dealer = Dealer::new(t, n, P_PLAINTEXT);
    let ks = dealer.generate_keypair::<PARAMS>();
    let dbl = dealer.generate_double_sharing(ell);

    let ek = ks.ek.clone();
    let hash = crypt::h(&ek.clone().serialize().into_vec());
    let z = random_32();

    let (_key_b, c) = mlkem::encaps::<PARAMS>(ek.clone());

    let parties: Vec<Party<{ PARAMS::K }>> =
        assemble_parties::<{ PARAMS::K }>(&ks.sk_shares, &dbl, dealer.thr);

    let mu_m_shares = threshold_decrypt(&parties, c.clone(), &dealer.thr);

    // ===== WARMUP =========================================================
    for _ in 0..TEE_WARMUP {
        let _ = tee_pipeline::<PARAMS>(&mu_m_shares, &dealer.thr, ek.clone(), hash, z, c.clone());
    }

    // ===== TIMED LOOP =====================================================
    let start = Instant::now();
    for _ in 0..TEE_ITERATIONS {
        let k = tee_pipeline::<PARAMS>(&mu_m_shares, &dealer.thr, ek.clone(), hash, z, c.clone());
        black_box(k);
    }
    let total = start.elapsed();
    total / TEE_ITERATIONS as u32
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

fn random_32() -> [u8; 32] {
    let mut rng = StdRng::from_entropy();
    let mut out = [0u8; 32];
    rng.fill_bytes(&mut out);
    out
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
