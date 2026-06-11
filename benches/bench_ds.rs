//! Bench: Ladon active (SPDZ2k) double sharing — single polynomial.
//!
//! Times the dealer's per-polynomial cost in the offline phase. One call to
//! `generate_double_sharing(1)` produces one double-shared polynomial: a
//! random polynomial d in R sampled coefficient-wise (uniform in [0, mu')),
//! then independently additively shared under both moduli (q' and mu')
//! across n committee members.
//!
//! The result is independent of the KEM parameter set (Ladon128 vs Ladon256),
//! since the polynomial lives in R_q regardless of module rank K. We sweep
//! committee size only.
//!
//! Convention (active, dishonest majority): t = n - 1 (all parties participate).
//!
//! Build & run:
//!     cargo bench --bench bench_double_sharing_active

#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use std::hint::black_box;
use std::time::{Duration, Instant};

use Ladon::dealer_spdz::DealerSpdz;
use Ladon::params::K_BITS;

// ===========================================================================
// ====== BENCH KNOBS — edit these ==========================================
// ===========================================================================
const N_LIST:     &[usize] = &[4, 8, 16, 32];
const S_BITS:     u32      = 40;
const P_PLAINTEXT: u128    = 2;

const ITERATIONS: usize = 500;
const WARMUP:     usize = 50;
// ===========================================================================

fn main() {
    println!("==================================================================");
    println!("  Ladon Bench: Active (SPDZ2k) Double-Shared Polynomial Generation");
    println!("==================================================================");
    println!();
    println!("  Per iteration:");
    println!("    Generate ONE double-shared polynomial d in R:");
    println!("      - sample d coefficient-wise (256 coefficients in [0, mu')),");
    println!("      - additively share each coefficient over Z_{{q'}},");
    println!("      - additively share each coefficient over Z_{{mu'}},");
    println!("      - regroup into per-party DoubleShare buffers.");
    println!();
    println!("  Convention: t = n - 1 (dishonest majority).");
    println!("  Result is independent of Ladon128/Ladon256 (one polynomial in R).");
    println!();
    println!("    k_bits / s_bits : {K_BITS} / {S_BITS}");
    println!("    p               : {P_PLAINTEXT}");
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
        bench_one(n);
    }
    println!("==================================================================");
}

fn bench_one(n: usize) {
    let t = n - 1;
    let dealer = DealerSpdz::new(n, K_BITS, S_BITS, P_PLAINTEXT);

    // Warmup.
    for _ in 0..WARMUP {
        let dbl = dealer.generate_double_sharing(1);
        black_box(dbl);
    }

    // Timed loop.
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let dbl = dealer.generate_double_sharing(1);
        black_box(dbl);
    }
    let total = start.elapsed();
    let avg = total / ITERATIONS as u32;
    let ops_per_sec = ITERATIONS as f64 / total.as_secs_f64();

    println!(
        "  {:>5}    {:>5}    {:>14}    {:>14}    {:>12.2}",
        t,
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