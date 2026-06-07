#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use std::hint::black_box;
use std::time::{Duration, Instant};
use Ladon::{kpke, params::*};

fn main() {
    let iterations: usize = 1000;
    println!("128-bit secure params: \n q: {} \n {:#?}", Q, Ladon128);
    bench_keygen::<Ladon128>("Ladon128", iterations)
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
    println!("=== PKE KeyGen benchmark: {label} ===");
    println!("iterations: {iterations}");

    // Warmup
    for _ in 0..100 {
        black_box(kpke::key_gen_2k::<PARAMS>());
    }

    let start = Instant::now();

    for _ in 0..iterations {
        black_box(kpke::key_gen_2k::<PARAMS>());
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