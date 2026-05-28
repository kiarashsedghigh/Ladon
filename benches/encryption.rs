#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use std::hint::black_box;
use std::time::{Duration, Instant};

use bitvec::view::BitView;
use rand::{rngs::StdRng, RngCore, SeedableRng};

use Moiragus::{kpke, params::*};
use Moiragus::ring::{Compressed, Ring};
use Moiragus::serialize::{BitOrder, MlKemDeserialize};

/// 32 cryptographically-random bytes. Mirrors `crypt::random_bytes::<32>()`,
/// reimplemented here because the `crypt` module is private to the crate.
fn random_32() -> [u8; 32] {
    let mut rng = StdRng::from_entropy();
    let mut out = [0u8; 32];
    rng.fill_bytes(&mut out);
    out
}
fn main() {
    let args: Vec<String> = std::env::args().collect();

    let paramset = args
        .iter()
        .map(|s| s.as_str())
        .find(|s| matches!(*s, "ML-KEM-512" | "ML-KEM-768" | "ML-KEM-1024"))
        .unwrap_or("ML-KEM-512");
    let iterations: usize = 1000;

    match paramset {
        "ML-KEM-512" => bench_encrypt::<MlKem512>("ML-KEM-512", iterations),
        "ML-KEM-768" => bench_encrypt::<MlKem768>("ML-KEM-768", iterations),
        "ML-KEM-1024" => bench_encrypt::<MlKem1024>("ML-KEM-1024", iterations),
        _ => panic!("Invalid parameter set: {paramset}"),
    }
}

fn bench_encrypt<PARAMS: MlKemParams>(label: &str, iterations: usize)
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
    println!("=== PKE Encrypt benchmark: {label} ===");
    println!("iterations: {iterations}");

    // KeyGen is setup only, not part of the timed encryption benchmark.
    let (ek_pke, _dk_pke) = kpke::key_gen::<PARAMS>();

    // Build one fixed 32-byte message as `Compressed<1, Ring>`.
    let m_bytes = random_32();
    let m: Compressed<1, Ring> =
        Compressed::<1, Ring>::deserialize(&m_bytes.view_bits::<BitOrder>().to_bitvec());

    // Fixed encryption randomness for benchmarking only.
    // This avoids benchmarking RNG cost.
    let r = random_32();

    // Warmup
    for _ in 0..100 {
        black_box(kpke::encrypt::<PARAMS>(
            black_box(ek_pke.clone()),
            black_box(m.clone()),
            black_box(r),
        ));
    }

    let start = Instant::now();

    for _ in 0..iterations {
        black_box(kpke::encrypt::<PARAMS>(
            black_box(ek_pke.clone()),
            black_box(m.clone()),
            black_box(r),
        ));
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