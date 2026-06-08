//! Bench: Active (SPDZ2k) key generation + lifted secret-key sharing only.
//!
//! Times the offline phase per (n) configuration:
//!   1. 2^k PKE keygen (kpke::key_gen_2k)
//!   2. Lifted additive sharing of the secret key over Z_{2^(k+s)} to n parties
//!
//! Double sharing is EXCLUDED — it can be precomputed independently and
//! benchmarked separately if needed; this bench focuses on the key-material
//! pipeline (mirrors the passive branch's bench_keygen_sharing).
//!
//! Threshold convention (active / dishonest-majority): t = n - 1.
//!
//! Build & run:
//!     cargo run --release --bin bench_keygen_sharing_active

#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use std::hint::black_box;
use std::time::{Duration, Instant};

use Ladon::additive_ring::share_vector_lifted;
use Ladon::additive_2k::SpdzParams;
use Ladon::kpke;
use Ladon::params::*;

// ===========================================================================
// ====== BENCH KNOBS — edit these ==========================================
// ===========================================================================
const N_LIST: &[usize] = &[4, 8, 16, 32];
const S_BITS: u32 = 40;
const P_PLAINTEXT: u128 = 2;
const ITERATIONS: usize = 100;
const WARMUP: usize = 20;
// ===========================================================================

fn main() {
    println!("==================================================================");
    println!("  Ladon Bench: Active (SPDZ2k) Keygen + Secret-Key Sharing");
    println!("==================================================================");
    println!();
    println!("  Per iteration:");
    println!("    1. 2^k PKE keygen (kpke::key_gen_2k).");
    println!("    2. Lifted additive secret-key sharing across n parties (Z_{{2^(k+s)}}).");
    println!();
    println!("  Double sharing is excluded; it can be precomputed and benched separately.");
    println!();
    println!("  Convention: t = n - 1 (dishonest majority).");
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
    [(); 32 * (PARAMS::D_U * PARAMS::K + PARAMS::D_V)]:,
{
    println!("------------------------------------------------------------------");
    println!("  {label}  (K = {})", PARAMS::K);
    println!("------------------------------------------------------------------");
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

    let thr = SpdzParams::new(K_BITS, S_BITS, P_PLAINTEXT);
    for &n in N_LIST {
        bench_one::<PARAMS>(n, &thr);
    }
}

fn bench_one<PARAMS: MlKemParams>(n: usize, thr: &SpdzParams)
where
    [(); 960 * PARAMS::K + 32]:,
    [(); 1920 * PARAMS::K + 96]:,
    [(); PARAMS::K]:,
    [(); PARAMS::ETA_1]:,
    [(); PARAMS::ETA_2]:,
    [(); 64 * PARAMS::ETA_1]:,
    [(); 32 * (PARAMS::D_U * PARAMS::K + PARAMS::D_V)]:,
{
    let t = n - 1;

    for _ in 0..WARMUP {
        let (ek, sk) = kpke::key_gen_2k::<PARAMS>();
        let sk_shares = share_vector_lifted::<{ PARAMS::K }>(&sk, n, thr);
        black_box((ek, sk_shares));
    }

    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let (ek, sk) = kpke::key_gen_2k::<PARAMS>();
        let sk_shares = share_vector_lifted::<{ PARAMS::K }>(&sk, n, thr);
        black_box((ek, sk_shares));
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