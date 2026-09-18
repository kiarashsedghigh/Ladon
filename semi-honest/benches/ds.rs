//! Bench: Ladon semi-honest double-sharing — single polynomial.
//!
//! Times the dealer's per-polynomial cost in the offline phase. One call to
//! `Dealer::generate_double_sharing(1)` produces ONE double-shared random
//! polynomial r in R: 256 coefficients sampled uniform in [0, mu'), then
//! ADDITIVELY shared under both moduli (q' and mu') across n parties.
//!
//! Why additive (not Shamir) for the double sharing:
//!   q' = 2^29 and mu' = q'/p = 2^28 are powers of two (not prime), so they
//!   are NOT fields and Shamir does not apply. The protocol gets around this
//!   because each party first converts its Shamir share of the secret key
//!   (over the prime F_q) into an additive share via local Lagrange at
//!   protocol time (Step 0 of paper Algorithm 2); from Step 1 onward
//!   everything is additive, and additive shares of r over composite Z_{q'}
//!   and Z_{mu'} compose correctly with that arithmetic.
//!
//! The result is independent of the KEM parameter set (Ladon128 vs Ladon256),
//! since the polynomial lives in R_q regardless of module rank K. We sweep
//! committee size only.
//!
//! Convention (semi-honest, honest majority): n = 2t + 1. THRESHOLDS lists
//! the min-to-decrypt t; the polynomial degree passed to Dealer is t - 1.
//!
//! Build & run:
//!     cargo bench --bench bench_double_sharing

#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use std::hint::black_box;
use std::time::{Duration, Instant};

use Ladon::dealer::Dealer;

// ===========================================================================
// ====== BENCH KNOBS — edit these ==========================================
// ===========================================================================
// t = min-to-decrypt. Committee size n = 2t + 1.
const THRESHOLDS: &[usize] = &[2, 4, 8, 16];
const P_PLAINTEXT: u64 = 2;

const ITERATIONS: usize = 500;
const WARMUP: usize = 50;
// ===========================================================================

fn main() {
    println!("==================================================================");
    println!("  Ladon Bench: Semi-Honest Double-Shared Polynomial Generation");
    println!("==================================================================");
    println!();
    println!("  Per iteration:");
    println!("    Generate ONE double-shared polynomial r in R via additive sharing:");
    println!("      - sample r coefficient-wise (256 coefficients in [0, mu')),");
    println!("      - additively share each coefficient over Z_{{q'}},");
    println!("      - additively share each coefficient over Z_{{mu'}},");
    println!("      - regroup into per-party DoubleShare buffers.");
    println!();
    println!("  Note: double sharing is ADDITIVE (not Shamir) because q' = 2^29");
    println!("        and mu' = 2^28 are powers of two, not prime.");
    println!();
    println!("  Convention: t = min-to-decrypt, n = 2t + 1 (honest majority).");
    println!("  Result is independent of Ladon128/Ladon256 (one polynomial in R).");
    println!();
    println!("    p           : {P_PLAINTEXT}");
    println!("    warmup/iter : {WARMUP} / {ITERATIONS}");
    println!();
    println!(
        "  {:>5}    {:>5}    {:>14}    {:>14}    {:>12}",
        "t", "n", "total", "avg/op", "ops/sec"
    );
    println!(
        "  {:>5}    {:>5}    {:>14}    {:>14}    {:>12}",
        "-----", "-----", "--------------", "--------------", "------------"
    );

    for &t in THRESHOLDS {
        let n = 2 * t + 1;
        bench_one(t, n);
    }
    println!("==================================================================");
}

fn bench_one(t: usize, n: usize) {
    // Dealer takes polynomial degree = t - 1 (any t shares reconstruct).
    let dealer = Dealer::new(t - 1, n, P_PLAINTEXT);

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
    } else if nanos >= 1_000 {
        format!("{:.3} us", nanos as f64 / 1_000.0)
    } else {
        format!("{nanos} ns")
    }
}