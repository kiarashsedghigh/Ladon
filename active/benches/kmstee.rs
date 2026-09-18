//! Bench: Active (SPDZ2k) Combined Committee + TEE Cost (per decryption request).
//!
//! This bench does NOT introduce any new measurement logic or parameters. It
//! simply RE-RUNS, unchanged, the two existing active-variant benches:
//!
//!   * committee_local        (benches/committee_local.rs)
//!       -> the per-party local compute a KBS committee member spends on one
//!          decryption request (avg / party-op). This cost is essentially
//!          n-independent, so it is measured once per security level using the
//!          same fixed convention as the source bench (SETUP_N).
//!
//!   * tee                    (benches/tee.rs)
//!       -> the TEE-side reconstruction + FO finalization performed after the
//!          committee has produced its shares (avg / op), swept over the same n.
//!
//! For each security level it prints one table. Per row (n in N_LIST, t = n-1):
//!
//!     committee/party   =  avg / party-op   (from committee_local.rs)
//!     tee/op            =  avg / op         (from tee.rs)
//!     total             =  committee/party  +  tee/op
//!
//! The two source constants, setups, warmup counts, iteration counts and timed
//! pipelines are copied verbatim below so the produced numbers match running
//! each source bench on its own; only the presentation (adding the two
//! columns) is new.
//!
//! Build & run:
//!     cargo bench --bench kmstee

#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use std::hint::black_box;
use std::time::{Duration, Instant};

use bitvec::view::BitView;
use rand::{rngs::StdRng, RngCore, SeedableRng};

use Ladon::crypt;
use Ladon::dealer_spdz::DealerSpdz;
use Ladon::kpke;
use Ladon::negacyclic::{decompress_ring_2k, decompress_vector_2k};
use Ladon::params::*;
use Ladon::party_spdz::PartySpdz;
use Ladon::ring::{Compressed, Ring};
use Ladon::serialize::{BitOrder, MlKemDeserialize, MlKemSerialize};
use Ladon::threshold_decrypt::{assemble_parties, open_e_tilde, receiver_reconstruct, threshold_decrypt};

// ===========================================================================
// ====== BENCH KNOBS — copied verbatim from the two source benches. ========
// ======  Do NOT edit: these must match committee_local.rs and tee.rs. =====
// ===========================================================================

// ell — parallel double sharings per security level.
// (Identical values in both source benches.)
const ELL_LADON128: usize = 5;
const ELL_LADON256: usize = 7;

const S_BITS: u32 = 36;
const P_PLAINTEXT: u128 = 2;

// ---- from tee.rs ----------------------------------------------------------
const N_LIST: &[usize] = &[4, 8, 16, 32];
const TEE_ITERATIONS: usize = 1500;
const TEE_WARMUP: usize = 200;

// ---- from committee_local.rs ---------------------------------------------
// Per-party local compute is essentially n-independent, so the source bench
// fixes a small n.
const SETUP_N: usize = 4;
const COMMITTEE_ITERATIONS: usize = 4000;
const COMMITTEE_WARMUP: usize = 200;
// ===========================================================================

fn main() {
    println!("==================================================================");
    println!("  Ladon Bench: Active (SPDZ2k) Combined Committee + TEE Cost");
    println!("==================================================================");
    println!();
    println!("  This bench re-runs, unchanged, two existing benches and prints");
    println!("  their per-op timings side by side plus their total:");
    println!();
    println!("    committee/party : committee_local avg / party-op");
    println!("                      (KBS member local compute for one request;");
    println!("                       essentially n-independent, measured once per");
    println!("                       security level at n = {SETUP_N}).");
    println!("    tee/op          : tee avg / op");
    println!("                      (TEE reconstruction + FO, swept over n).");
    println!("    total           : committee/party + tee/op.");
    println!();
    println!("  No source parameters or timed logic were modified; only the two");
    println!("  numbers are combined into one table.");
    println!();
    println!("  Convention: t = n - 1 (dishonest majority).");
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
    println!("  {label}  (K = {}, ell = {ell})", PARAMS::K);
    println!("------------------------------------------------------------------");
    println!("    k_bits / s_bits          : {K_BITS} / {S_BITS}");
    println!("    p                        : {P_PLAINTEXT}");
    println!(
        "    committee n (fixed)      : {SETUP_N}    (n-independent, measured once)"
    );
    println!(
        "    committee warmup / iter  : {COMMITTEE_WARMUP} / {COMMITTEE_ITERATIONS}"
    );
    println!("    tee warmup / iter        : {TEE_WARMUP} / {TEE_ITERATIONS}");
    println!();

    // ---- Committee: measured once (per-party cost is n-independent) -------
    let committee_avg = measure_committee::<PARAMS>(ell);

    println!(
        "  {:>5}    {:>5}    {:>16}    {:>16}    {:>16}",
        "t", "n", "committee/party", "tee/op", "total"
    );
    println!(
        "  {:>5}    {:>5}    {:>16}    {:>16}    {:>16}",
        "-----", "-----", "----------------", "----------------", "----------------"
    );

    for &n in N_LIST {
        let tee_avg = measure_tee::<PARAMS>(n, ell);
        let total = committee_avg + tee_avg;
        println!(
            "  {:>5}    {:>5}    {:>16}    {:>16}    {:>16}",
            n - 1,
            n,
            format_duration(committee_avg),
            format_duration(tee_avg),
            format_duration(total)
        );
    }
}

// ===========================================================================
// ====== Committee measurement — copied verbatim from committee_local.rs ===
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
    // ----- SETUP (untimed) -------------------------------------------------
    let dealer = DealerSpdz::new(SETUP_N, K_BITS, S_BITS, P_PLAINTEXT);
    let ks = dealer.generate_keypair::<PARAMS>();
    let dbl = dealer.generate_double_sharing(ell);

    let mut rng = StdRng::from_entropy();
    let mut msg_bytes = [0u8; 32];
    rng.fill_bytes(&mut msg_bytes);
    let m: Compressed<1, Ring> =
        Compressed::<1, Ring>::deserialize(&msg_bytes.view_bits::<BitOrder>().to_bitvec());
    let mut rand = [0u8; 32];
    rng.fill_bytes(&mut rand);
    let ct = kpke::encrypt_2k::<PARAMS>(ks.ek.clone(), m, rand);
    let u_dec = decompress_vector_2k::<{ PARAMS::K }>(&ct.0 .0, PARAMS::D_U as u32, K_BITS);
    let v_dec = decompress_ring_2k(&ct.1 .0, PARAMS::D_V as u32, K_BITS);

    let parties: Vec<PartySpdz<{ PARAMS::K }>> =
        assemble_parties::<{ PARAMS::K }>(&ks.sk_shares, &dbl, dealer.thr);
    let v_holder_id = parties.iter().map(|p| p.id).min().unwrap();

    // Precompute the opened e_tilde_j vectors once (network round, untimed).
    let step1_all: Vec<_> = parties
        .iter()
        .map(|p| p.partial_decrypt(&u_dec, &v_dec, p.id == v_holder_id))
        .collect();
    let masked_all: Vec<Vec<[u128; 256]>> = parties
        .iter()
        .zip(step1_all.iter())
        .map(|(p, out)| p.mask(&out.e_prime))
        .collect();
    let e_tilde_opened = open_e_tilde(&masked_all, &dealer.thr);

    // Sanity: protocol works end to end at this ELL.
    {
        let mu_m_shares = threshold_decrypt(&parties, &u_dec, &v_dec, &dealer.thr);
        assert_eq!(mu_m_shares[0].len(), ell);
        black_box(mu_m_shares);
    }

    // We time a non-holder party (the common case).
    let target = &parties[1];
    let is_holder = target.id == v_holder_id;
    debug_assert!(!is_holder);

    // ----- WARMUP ----------------------------------------------------------
    for _ in 0..COMMITTEE_WARMUP {
        black_box(committee_pipeline::<PARAMS>(target, &u_dec, &v_dec, &e_tilde_opened, is_holder));
    }

    // ----- TIMED -----------------------------------------------------------
    let start = Instant::now();
    for _ in 0..COMMITTEE_ITERATIONS {
        black_box(committee_pipeline::<PARAMS>(target, &u_dec, &v_dec, &e_tilde_opened, is_holder));
    }
    let total = start.elapsed();
    total / COMMITTEE_ITERATIONS as u32
}

#[inline(never)]
fn committee_pipeline<PARAMS: MlKemParams>(
    party: &PartySpdz<{ PARAMS::K }>,
    u: &Ladon::ring::Vector<{ PARAMS::K }>,
    v: &Ring,
    e_tilde_opened: &[[u128; 256]],
    is_holder: bool,
) -> Vec<[u128; 256]>
where
    [(); PARAMS::K]:,
{
    let step1 = party.partial_decrypt(u, v, is_holder);
    let masked = party.mask(&step1.e_prime);
    black_box(&masked);
    party.finalize(&step1.w_prime, e_tilde_opened, is_holder)
}

// ===========================================================================
// ====== TEE measurement — copied verbatim from tee.rs =====================
// ====== (returns avg / op instead of printing) =============================
// ===========================================================================
fn measure_tee<PARAMS: MlKemParams>(n: usize, ell: usize) -> Duration
where
    [(); 960 * PARAMS::K + 32]:,
    [(); 1920 * PARAMS::K + 96]:,
    [(); PARAMS::K]:,
    [(); PARAMS::ETA_1]:,
    [(); PARAMS::ETA_2]:,
    [(); 64 * PARAMS::ETA_1]:,
    [(); 64 * PARAMS::ETA_2]:,
    [(); 32 * (PARAMS::D_U * PARAMS::K + PARAMS::D_V)]:,
    [(); 960 * PARAMS::K + 32]:,
{
    // ----- SETUP (untimed) -------------------------------------------------
    let dealer = DealerSpdz::new(n, K_BITS, S_BITS, P_PLAINTEXT);
    let ks = dealer.generate_keypair::<PARAMS>();
    let dbl = dealer.generate_double_sharing(ell);

    // Asset-owner side: build a real ciphertext using the FO encaps path.
    let z = crypt::random_bytes::<32>();
    let ek_hash = crypt::h(&ks.ek.serialize().into_vec());

    let mut rng = StdRng::from_entropy();
    let mut m_bytes = [0u8; 32];
    rng.fill_bytes(&mut m_bytes);
    let mut combined = [0u8; 64];
    combined[..32].copy_from_slice(&m_bytes);
    combined[32..].copy_from_slice(&ek_hash);
    let (_key_a, rand) = crypt::g::<64>(&combined);

    let m: Compressed<1, Ring> =
        Compressed::<1, Ring>::deserialize(&m_bytes.view_bits::<BitOrder>().to_bitvec());
    let ct = kpke::encrypt_2k::<PARAMS>(ks.ek.clone(), m, rand);

    // Committee side (untimed): produce <mu'*m>^{q'} shares per party.
    let u_dec = decompress_vector_2k::<{ PARAMS::K }>(&ct.0 .0, PARAMS::D_U as u32, K_BITS);
    let v_dec = decompress_ring_2k(&ct.1 .0, PARAMS::D_V as u32, K_BITS);
    let parties: Vec<PartySpdz<{ PARAMS::K }>> =
        assemble_parties::<{ PARAMS::K }>(&ks.sk_shares, &dbl, dealer.thr);
    let mu_m_shares = threshold_decrypt(&parties, &u_dec, &v_dec, &dealer.thr);

    // ----- WARMUP ----------------------------------------------------------
    for _ in 0..TEE_WARMUP {
        let key = tee_pipeline::<PARAMS>(&mu_m_shares, &dealer.thr, &ks.ek, &ek_hash, &z, &ct);
        black_box(key);
    }

    // ----- TIMED -----------------------------------------------------------
    let start = Instant::now();
    for _ in 0..TEE_ITERATIONS {
        let key = tee_pipeline::<PARAMS>(&mu_m_shares, &dealer.thr, &ks.ek, &ek_hash, &z, &ct);
        black_box(key);
    }
    let total = start.elapsed();
    total / TEE_ITERATIONS as u32
}

#[inline(never)]
fn tee_pipeline<PARAMS: MlKemParams>(
    mu_m_shares: &[Vec<[u128; 256]>],
    thr: &Ladon::additive_2k::SpdzParams,
    ek: &kpke::KpkeEncryptionKey<{ PARAMS::K }>,
    ek_hash: &[u8; 32],
    z: &[u8; 32],
    ct: &kpke::Cyphertext<{ PARAMS::K }, { PARAMS::D_U }, { PARAMS::D_V }>,
) -> [u8; 32]
where
    [(); PARAMS::K]:,
    [(); PARAMS::ETA_1]:,
    [(); PARAMS::ETA_2]:,
    [(); 64 * PARAMS::ETA_1]:,
    [(); 64 * PARAMS::ETA_2]:,
    [(); 32 * (PARAMS::D_U * PARAMS::K + PARAMS::D_V)]:,
    [(); 960 * PARAMS::K + 32]:,
{
    // Step 6: reconstruct m (with majority over ell).
    let m_bytes = receiver_reconstruct(mu_m_shares, thr);

    // FO finalization.
    let mut combined = [0u8; 64];
    combined[..32].copy_from_slice(&m_bytes);
    combined[32..].copy_from_slice(ek_hash);
    let (key_candidate, rand_prime) = crypt::g::<64>(&combined);

    let m: Compressed<1, Ring> =
        Compressed::<1, Ring>::deserialize(&m_bytes.view_bits::<BitOrder>().to_bitvec());
    let ct_prime = kpke::encrypt_2k::<PARAMS>(ek.clone(), m, rand_prime);

    if ct_prime.0 == ct.0 && ct_prime.1 == ct.1 {
        key_candidate
    } else {
        crypt::j([&z[..], ct.serialize().as_raw_slice()].concat())
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
