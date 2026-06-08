//! Bench: Active (SPDZ2k) TEE-side reconstruction.
//!
//! Times receiver_reconstruct (Step 6): sum the n parties' <mu'*m>^{q'}
//! shares per j, decode each ell-candidate to a bit, majority-vote per
//! coefficient, pack into 32 bytes.
//!
//! Note: the active branch is currently PKE-level. A KEM-level wrapper
//! (with FO finalization via SHA3) would add ~hundreds of microseconds
//! constant overhead; that's not included here.
//!
//! Sweeps n in {4, 8, 16, 32} (with t = n - 1 in active). The committee-size
//! term in receiver_reconstruct is just the inner summation cost; it grows
//! linearly in n.
//!
//! Build & run:
//!     cargo run --release --bin bench_tee_active

#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use std::hint::black_box;
use std::time::{Duration, Instant};

use Ladon::dealer_spdz::DealerSpdz;
use Ladon::kpke;
use Ladon::params::*;
use Ladon::party_spdz::PartySpdz;
use Ladon::threshold_decrypt::{assemble_parties, receiver_reconstruct, threshold_decrypt};
use Ladon::ring::{Compressed, Ring};
use Ladon::serialize::{BitOrder, MlKemDeserialize};
use Ladon::negacyclic::{decompress_ring_2k, decompress_vector_2k};
use bitvec::view::BitView;
use rand::{rngs::StdRng, RngCore, SeedableRng};

// ===========================================================================
// ====== BENCH KNOBS — edit these ==========================================
// ===========================================================================
const N_LIST: &[usize] = &[4, 8, 16, 32];
const ELL: usize = 5;
const K_BITS: u32 = 30;
const S_BITS: u32 = 40;
const P_PLAINTEXT: u128 = 2;
const ITERATIONS: usize = 1500;
const WARMUP: usize = 200;
// ===========================================================================

fn main() {
    println!("==================================================================");
    println!("  Ladon Bench: Active (SPDZ2k) TEE-Side Receiver Reconstruction");
    println!("==================================================================");
    println!();
    println!("  Times receiver_reconstruct (Step 6): sum per-party <mu'*m>^{{q'}}");
    println!("  shares per j, decode ell candidates, majority-vote per coefficient,");
    println!("  pack to 32 bytes. PKE-level only; FO finalization would add a small");
    println!("  constant if a KEM-level wrapper is in use.");
    println!();
    println!("  Convention: t = n - 1 (dishonest majority).");
    println!();
    run::<MlKem512>("MlKem512 (K=6)");
    println!("==================================================================");
}

fn run<PARAMS: MlKemParams>(label: &str)
where
    [(); 384 * PARAMS::K + 32]:,
    [(); 768 * PARAMS::K + 96]:,
    [(); PARAMS::K]:,
    [(); PARAMS::ETA_1]:,
    [(); PARAMS::ETA_2]:,
    [(); 64 * PARAMS::ETA_1]:,
    [(); 64 * PARAMS::ETA_2]:,
    [(); 32 * (PARAMS::D_U * PARAMS::K + PARAMS::D_V)]:,
{
    println!("------------------------------------------------------------------");
    println!("  {label}");
    println!("------------------------------------------------------------------");
    println!("    K               : {}", PARAMS::K);
    println!("    k_bits / s_bits : {K_BITS} / {S_BITS}");
    println!("    p               : {P_PLAINTEXT}");
    println!("    ell             : {ELL}");
    println!("    warmup / iter   : {WARMUP} / {ITERATIONS}");
    println!();
    println!(
        "  {:>5}    {:>5}    {:>14}    {:>14}    {:>12}",
        "t", "n", "total", "avg/op", "ops/sec"
    );
    println!(
        "  {:>5}    {:>5}    {:>14}    {:>14}    {:>12}",
        "-----", "-----", "--------------", "--------------", "------------"
    );

    for &n in N_LIST {
        bench_one::<PARAMS>(n);
    }
}

fn bench_one<PARAMS: MlKemParams>(n: usize)
where
    [(); 384 * PARAMS::K + 32]:,
    [(); 768 * PARAMS::K + 96]:,
    [(); PARAMS::K]:,
    [(); PARAMS::ETA_1]:,
    [(); PARAMS::ETA_2]:,
    [(); 64 * PARAMS::ETA_1]:,
    [(); 64 * PARAMS::ETA_2]:,
    [(); 32 * (PARAMS::D_U * PARAMS::K + PARAMS::D_V)]:,
{
    // ----- SETUP (untimed): run the committee once to get its outputs ------
    let dealer = DealerSpdz::new(n, K_BITS, S_BITS, P_PLAINTEXT);
    let ks = dealer.generate_keypair::<PARAMS>();
    let dbl = dealer.generate_double_sharing(ELL);

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
    let mu_m_shares = threshold_decrypt(&parties, &u_dec, &v_dec, &dealer.thr);

    // ----- WARMUP ----------------------------------------------------------
    for _ in 0..WARMUP {
        let r = receiver_reconstruct(&mu_m_shares, &dealer.thr);
        black_box(r);
    }

    // ----- TIMED -----------------------------------------------------------
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let r = receiver_reconstruct(&mu_m_shares, &dealer.thr);
        black_box(r);
    }
    let total = start.elapsed();
    let avg = total / ITERATIONS as u32;
    let ops_per_sec = ITERATIONS as f64 / total.as_secs_f64();

    println!(
        "  {:>5}    {:>5}    {:>14}    {:>14}    {:>12.2}",
        n - 1,
        n,
        format_duration(total),
        format_duration(avg),
        ops_per_sec,
    );
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