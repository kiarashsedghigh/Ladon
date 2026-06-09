//! Bench: Active (SPDZ2k) per-party committee compute.
//!
//! Times the local compute one KBS party performs during a single threshold
//! decryption (Steps 1-5 minus the open-broadcast round, which is network).
//!
//! Per party per iteration:
//!   1. partial_decrypt — linear inner product against the lifted key share,
//!      mod-switch q -> q' (identity since q = 2^k), isolate noise mod mu'.
//!   2. mask            — add ell shares of d_j to <e'>^{mu'}.
//!   3. finalize        — compute the party's ell shares of (mu' * m)_j over Z_{q'}.
//!
//! EXCLUDED (network rounds):
//!   - broadcast/open of e_tilde_j,
//!   - sending final shares to the TEE.
//!
//! Per-paramset ell (matches the demo and bench_tee).
//!
//! Convention: t = n - 1. SETUP_N is fixed because per-party local compute
//! is essentially n-independent.
//!
//! Build & run:
//!     cargo run --release --bin bench_committee_local_active

#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use std::hint::black_box;
use std::time::{Duration, Instant};

use bitvec::view::BitView;
use rand::{rngs::StdRng, RngCore, SeedableRng};

use Ladon::dealer_spdz::DealerSpdz;
use Ladon::kpke;
use Ladon::negacyclic::{decompress_ring_2k, decompress_vector_2k};
use Ladon::params::*;
use Ladon::party_spdz::PartySpdz;
use Ladon::ring::{Compressed, Ring};
use Ladon::serialize::{BitOrder, MlKemDeserialize};
use Ladon::threshold_decrypt::{assemble_parties, open_e_tilde, threshold_decrypt};

// ===========================================================================
// ====== BENCH KNOBS — edit these ==========================================
// ===========================================================================
const ELL_LADON128: usize = 5;
const ELL_LADON256: usize = 7;
const S_BITS: u32 = 36;
const P_PLAINTEXT: u128 = 2;
// Per-party local compute is essentially n-independent; pick a small n.
const SETUP_N: usize = 4;
const ITERATIONS: usize = 1500;
const WARMUP: usize = 200;
// ===========================================================================

fn main() {
    println!("==================================================================");
    println!("  Ladon Bench: Active (SPDZ2k) Committee Local Compute (per-party)");
    println!("==================================================================");
    println!();
    println!("  Pipeline per iteration (one party):");
    println!("    1. partial_decrypt (Steps 1-3): linear inner product, mod switch,");
    println!("       isolate error.");
    println!("    2. mask              (Step 4 local): add ell d_j shares.");
    println!("    3. finalize          (Step 5): compute ell <mu'*m>^{{q'}} shares.");
    println!();
    println!("  EXCLUDED:");
    println!("    - broadcast/open of e_tilde_j (network),");
    println!("    - sending final shares to TEE (network).");
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
    println!("    k_bits / s_bits : {K_BITS} / {S_BITS}");
    println!("    p               : {P_PLAINTEXT}");
    println!("    n (fixed)       : {SETUP_N}    (per-party compute is n-independent)");
    println!("    warmup / iter   : {WARMUP} / {ITERATIONS}");
    println!();

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
    for _ in 0..WARMUP {
        black_box(local_pipeline::<PARAMS>(target, &u_dec, &v_dec, &e_tilde_opened, is_holder));
    }

    // ----- TIMED -----------------------------------------------------------
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        black_box(local_pipeline::<PARAMS>(target, &u_dec, &v_dec, &e_tilde_opened, is_holder));
    }
    let total = start.elapsed();
    let avg = total / ITERATIONS as u32;
    let ops_per_sec = ITERATIONS as f64 / total.as_secs_f64();

    println!("    total time     : {}", format_duration(total));
    println!("    avg / party-op : {}", format_duration(avg));
    println!("    ops / sec      : {ops_per_sec:.2}");
    println!();
    println!("  Multiply avg/op by t = {} for the aggregate sequential committee compute.", SETUP_N - 1);
}

#[inline(never)]
fn local_pipeline<PARAMS: MlKemParams>(
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