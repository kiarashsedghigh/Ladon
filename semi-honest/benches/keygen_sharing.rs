//! Bench: Ladon KMS Committee Provision (Algorithm 1 of the paper).
//!
//! Times the dealer's full offline keygen + Shamir-share-the-secret-key
//! pipeline (`Dealer::generate_keypair`).
//!
//! Threshold convention:
//!   t = THRESHOLD = the minimum size of a decrypting set.
//!     - any t (or more) parties can decrypt.
//!     - any t-1 cannot.
//!   Shamir polynomial degree = t-1, so Dealer::new takes t-1.
//!   Committee size n = 2t + 1 (smallest n satisfying t < n/2).
//!
//! Build & run:
//!     cargo run --release --bin bench_keygen_sharing

#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use std::hint::black_box;
use std::time::{Duration, Instant};

use Ladon::dealer::Dealer;
use Ladon::params::*;

// ===========================================================================
// ====== BENCH KNOBS — edit these ==========================================
// ===========================================================================
const THRESHOLDS: &[usize] = &[4, 8, 16, 32];
const P_PLAINTEXT: u64 = 2;
const ITERATIONS: usize = 1000;
const WARMUP: usize = 100;
// ===========================================================================

fn main() {
    println!("==================================================================");
    println!("  Ladon Bench: KMS Committee Provision (keygen + Shamir sharing)");
    println!("==================================================================");
    println!();
    println!("  What this bench measures:");
    println!("    Time for the dealer to generate the public key (ek) and the");
    println!("    underlying secret key dk, then Shamir-share dk across n parties");
    println!("    so any t (or more) can cooperate to decrypt. This is Algorithm");
    println!("    1 of the paper - it runs ONCE per committee provisioning.");
    println!();
    println!("    Dominant cost: Shamir polynomial sampling at degree t-1, on");
    println!("    each of the K rings (= K x 256 polynomial coefficients shared).");
    println!();
    println!("  Convention: t = min-to-decrypt. n = 2t + 1.");
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
    println!("    p     (plaintext modulus)   : {P_PLAINTEXT}");
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
        bench_one::<PARAMS>(t, n);
    }
}

fn bench_one<PARAMS: MlKemParams>(t: usize, n: usize)
where
    [(); 960 * PARAMS::K + 32]:,
    [(); 1920 * PARAMS::K + 96]:,
    [(); PARAMS::K]:,
    [(); PARAMS::ETA_2]:,
    [(); 64 * PARAMS::ETA_1]:,
    [(); 32 * (PARAMS::D_U * PARAMS::K + PARAMS::D_V)]:,
{
    // t = min-to-decrypt; polynomial degree = t-1.
    let dealer = Dealer::new(t - 1, n, P_PLAINTEXT);

    for _ in 0..WARMUP {
        black_box(dealer.generate_keypair::<PARAMS>());
    }

    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let ks = dealer.generate_keypair::<PARAMS>();
        black_box(ks);
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

fn format_duration(d: Duration) -> String {
    // Always report in milliseconds so every timing field shares one unit.
    format!("{:.3} ms", d.as_nanos() as f64 / 1_000_000.0)
}