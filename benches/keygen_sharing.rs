#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use std::hint::black_box;
use std::time::{Duration, Instant};

use Ladon::{kpke, params::*};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let paramset = args
        .iter()
        .map(|s| s.as_str())
        .find(|s| matches!(*s, "ML-KEM-512" | "ML-KEM-768" | "ML-KEM-1024"))
        .unwrap_or("ML-KEM-512");
    let iterations: usize = 1000;


    match paramset {
        "ML-KEM-512" => bench_keygen::<MlKem512>("ML-KEM-512", iterations),
        "ML-KEM-768" => bench_keygen::<MlKem768>("ML-KEM-768", iterations),
        "ML-KEM-1024" => bench_keygen::<MlKem1024>("ML-KEM-1024", iterations),
        _ => panic!("Invalid parameter set: {paramset}"),
    }
}

fn bench_keygen<PARAMS: MlKemParams>(label: &str, iterations: usize)
where
    [(); 384 * PARAMS::K + 32]:,
    [(); 768 * PARAMS::K + 96]:,
    [(); PARAMS::K]:,
    [(); PARAMS::ETA_1]:,
    [(); PARAMS::ETA_2]:,
    [(); PARAMS::D_U]:,
    [(); 64 * PARAMS::ETA_1]:,
    [(); 64 * PARAMS::ETA_2]:,
    [(); 32 * (PARAMS::D_U * PARAMS::K + PARAMS::D_V)]:,
{
    println!("=== K-PKE KeyGen benchmark: {label} ===");
    println!("iterations: {iterations}");

    // Warmup
    for _ in 0..100 {
        black_box(kpke::key_gen::<PARAMS>());
    }

    let start = Instant::now();

    for _ in 0..iterations {
        black_box(kpke::key_gen::<PARAMS>());
    }

    let total = start.elapsed();
    let avg = total / iterations as u32;

    println!("total:   {}", format_duration(total));
    println!("average: {}", format_duration(avg));
    println!("ops/sec: {:.2}", iterations as f64 / total.as_secs_f64());
}

fn format_duration(d: Duration) -> String {
    let nanos = d.as_nanos();

    if nanos >= 1_000_000_000 {
        format!("{:.6} s", nanos as f64 / 1_000_000_000.0)
    } else if nanos >= 1_000_000 {
        format!("{:.6} ms", nanos as f64 / 1_000_000.0)
    } else if nanos >= 1_000 {
        format!("{:.6} us", nanos as f64 / 1_000.0)
    } else {
        format!("{nanos} ns")
    }
}