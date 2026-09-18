//! Bench: Ladon Owner Asset Key Wrapping (Algorithm 2 of the paper).
//!
//! Times the asset owner's encapsulation pipeline: K-PKE-encrypt a fresh
//! 32-byte seed, then FO-derive the asset key. Independent of the committee
//! size n and threshold t (Section 4.2 of the paper), so this bench reports
//! a single number per parameter set.
//!
//! Build & run:
//!     cargo run --release --bin bench_encapsulation

#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use std::hint::black_box;
use std::time::{Duration, Instant};

use Ladon::dealer::Dealer;
use Ladon::mlkem;
use Ladon::params::*;

// ===========================================================================
// ====== BENCH KNOBS — edit these ==========================================
// ===========================================================================
const P_PLAINTEXT: u64 = 2;
// Threshold convention: t = min-to-decrypt. Polynomial degree = t-1.
// These knobs only feed the dealer (offline setup); encaps is independent
// of t and n.
const SETUP_T: usize = 4;
const SETUP_N: usize = 9;        // satisfies SETUP_T < SETUP_N / 2
const ITERATIONS: usize = 5000;
const WARMUP: usize = 100;
// ===========================================================================

fn main() {
    println!("==================================================================");
    println!("  Ladon Bench: Owner Asset Key Wrapping (encaps)");
    println!("==================================================================");
    println!();
    println!("  What this bench measures:");
    println!("    Time for the asset owner to:");
    println!("      1. Sample a fresh 32-byte secret seed.");
    println!("      2. K-PKE-encrypt the seed under ek (matrix sample, NTT");
    println!("         multiplies, compression).");
    println!("      3. FO-derive the asset key K_B = G(seed, H(ek)).");
    println!("    The result is (K_B, ciphertext). The ciphertext is sent to the");
    println!("    TEE for decapsulation by the committee.");
    println!();
    println!("    This operation does NOT depend on the committee size n or the");
    println!("    threshold t. The dealer is set up only to obtain a valid ek.");
    println!();
    run::<Ladon128>("Ladon128");
    println!();
    run::<Ladon256>("Ladon256");
    println!("==================================================================");
}

fn run<PARAMS: MlKemParams>(label: &str)
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
    println!("    warmup / timed iterations   : {WARMUP} / {ITERATIONS}");
    println!();

    // ---- Offline setup (NOT timed) ---------------------------------------
    // SETUP_T = min-to-decrypt; polynomial degree = SETUP_T - 1.
    let dealer = Dealer::new(SETUP_T - 1, SETUP_N, P_PLAINTEXT);
    let ks = dealer.generate_keypair::<PARAMS>();
    let ek = ks.ek.clone();

    // ---- Warmup ----------------------------------------------------------
    for _ in 0..WARMUP {
        let (key, c) = mlkem::encaps::<PARAMS>(black_box(ek.clone()));
        black_box((key, c));
    }

    // ---- Timed loop ------------------------------------------------------
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let (key, c) = mlkem::encaps::<PARAMS>(black_box(ek.clone()));
        black_box((key, c));
    }
    let total = start.elapsed();
    let avg = total / ITERATIONS as u32;
    let ops_per_sec = ITERATIONS as f64 / total.as_secs_f64();

    println!("    total time    : {}", format_duration(total));
    println!("    avg / encaps  : {}", format_duration(avg));
    println!("    ops / sec     : {:.2}", ops_per_sec);
}

fn format_duration(d: Duration) -> String {
    // Always report in milliseconds so every timing field shares one unit.
    format!("{:.3} ms", d.as_nanos() as f64 / 1_000_000.0)
}