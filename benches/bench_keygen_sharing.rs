//! Bench: Active (SPDZ2k) key generation + secret-key sharing + double sharing.
//!
//! Times the offline phase per (n) configuration:
//!   - 2^k PKE keygen (kpke::key_gen_2k)
//!   - Lifted additive sharing of the secret key over Z_{2^(k+s)} to n parties
//!   - One round of double sharing of length ELL (over q' and mu')
//!
//! Threshold convention (active / dishonest-majority): t = n - 1. ALL n
//! parties must participate at decryption time; the bench just sweeps over
//! committee sizes n.
//!
//! Build & run:
//!     cargo run --release --bin bench_keygen_sharing_active

#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use std::hint::black_box;
use std::time::{Duration, Instant};

use Ladon::dealer_spdz::DealerSpdz;
use Ladon::params::*;

// ===========================================================================
// ====== BENCH KNOBS — edit these ==========================================
// ===========================================================================
const N_LIST: &[usize] = &[4, 8, 16, 32];
const K_BITS: u32 = 30;
const S_BITS: u32 = 40;
const P_PLAINTEXT: u128 = 2;
const ELL: usize = 5;
const ITERATIONS: usize = 100;
const WARMUP: usize = 20;
// ===========================================================================

fn main() {
    println!("==================================================================");
    println!("  Ladon Bench: Active (SPDZ2k) Keygen + Secret-Key + Double Sharing");
    println!("==================================================================");
    println!();
    println!("  Offline phase per iteration:");
    println!("    1. 2^k PKE keygen.");
    println!("    2. Additive secret-key sharing (lifted to Z_{{2^(k+s)}}) across n parties.");
    println!("    3. ell = {ELL} parallel double sharings over (q', mu').");
    println!();
    println!("  Convention: t = n - 1 (dishonest majority; all n participate at decryption).");
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
    [(); 32 * (PARAMS::D_U * PARAMS::K + PARAMS::D_V)]:,
{
    println!("------------------------------------------------------------------");
    println!("  {label}");
    println!("------------------------------------------------------------------");
    println!("    K           : {}", PARAMS::K);
    println!("    k_bits      : {K_BITS}    (q = 2^k)");
    println!("    s_bits      : {S_BITS}   (lift width)");
    println!("    p           : {P_PLAINTEXT}");
    println!("    ell         : {ELL}");
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
    [(); 32 * (PARAMS::D_U * PARAMS::K + PARAMS::D_V)]:,
{
    let t = n - 1;

    // Warmup
    for _ in 0..WARMUP {
        let dealer = DealerSpdz::new(n, K_BITS, S_BITS, P_PLAINTEXT);
        let ks = dealer.generate_keypair::<PARAMS>();
        let dbl = dealer.generate_double_sharing(ELL);
        black_box((ks, dbl));
    }

    // Timed
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let dealer = DealerSpdz::new(n, K_BITS, S_BITS, P_PLAINTEXT);
        let ks = dealer.generate_keypair::<PARAMS>();
        let dbl = dealer.generate_double_sharing(ELL);
        black_box((ks, dbl));
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