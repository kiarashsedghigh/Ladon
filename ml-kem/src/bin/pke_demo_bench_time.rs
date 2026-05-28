//! K-PKE round-trip demo + microbenchmark.
//! Put this at `ml-kem/src/bin/pke_demo.rs`.
//!
//! Run with:
//!     cargo run --release --bin pke_demo
//!
//! USE --release. Numbers in debug mode are not meaningful: the polynomial
//! arithmetic relies on the optimizer inlining `barrett_reduce` and the
//! NTT butterflies; without -O the figures are 10-50x slower than reality.
//!
//! Requires three small library edits:
//!   * src/lib.rs:    `mod pke;`   -> `pub mod pke;`
//!   * src/lib.rs:    `mod param;` -> `pub mod param;`
//!   * src/pke.rs:    `pub(crate)` -> `pub` on
//!                      DecryptionKey, EncryptionKey, generate, encrypt, decrypt
//!
//! What this does:
//!   Phase 1 (correctness): for each of MlKem{512,768,1024}, run a single
//!       KeyGen -> Encrypt -> Decrypt and check m' == m. Same as
//!       `round_trip_test` in pke.rs.
//!   Phase 2 (benchmark): for each parameter set, time KeyGen, Encrypt,
//!       and Decrypt separately, after warmup. Reports min / median / mean
//!       / max in microseconds, plus operations per second.
//!
//! Caveats on the numbers:
//!   - Encrypt re-derives the public matrix A_hat from rho on every call
//!     (a SHAKE128 expansion of K^2 polynomials). That's what the library
//!     does and the K-PKE.Encrypt spec assumes. If a real protocol caches
//!     A_hat per encryption key, encryption would be a lot cheaper than
//!     what you see here.
//!   - Decrypt has no SHAKE work and so is much faster than Encrypt.
//!   - `std::black_box` is used to keep the optimizer from realizing that
//!     the results are unused and deleting the calls entirely.
//!   - This is a single-threaded wall-clock measurement, not Criterion.
//!     No outlier removal, no statistical confidence intervals. Treat the
//!     median as the headline number; the spread tells you about system
//!     noise (OS scheduling, frequency scaling, etc.).

#![allow(non_snake_case)]

use std::hint::black_box;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use ml_kem::param::PkeParams;
use ml_kem::pke::{DecryptionKey, EncryptionKey};
use ml_kem::{B32, MlKem512, MlKem768, MlKem1024};

// ---------------------------------------------------------------------------
// RNG (not cryptographic — used only to make each run's inputs distinct).
// ---------------------------------------------------------------------------

struct SplitMix64(u64);
impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn fill(&mut self, out: &mut [u8]) {
        for chunk in out.chunks_mut(8) {
            let bytes = self.next_u64().to_le_bytes();
            chunk.copy_from_slice(&bytes[..chunk.len()]);
        }
    }
}
fn seed_from_time() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0xDEAD_BEEF_CAFE_BABE)
}
fn rand_b32(rng: &mut SplitMix64) -> B32 {
    let mut out = B32::default();
    rng.fill(out.as_mut_slice());
    out
}

// ---------------------------------------------------------------------------
// Phase 1: correctness round-trip.
// ---------------------------------------------------------------------------

fn round_trip<P: PkeParams>(label: &str, rng: &mut SplitMix64) {
    let d = rand_b32(rng);
    let (dk, ek): (DecryptionKey<P>, EncryptionKey<P>) = DecryptionKey::<P>::generate(&d);

    let msg: B32 = rand_b32(rng);
    let r:   B32 = rand_b32(rng);
    let ct = ek.encrypt(&msg, &r);

    let recovered: B32 = dk.decrypt(&ct);

    print!("[{label:>9}] msg = {}... ct = {} bytes ... ", hex_prefix(&msg, 8), ct.len());
    if recovered == msg {
        println!("OK");
    } else {
        println!("MISMATCH");
        println!("           got = {}", hex_prefix(&recovered, 32));
        std::process::exit(1);
    }
}

// ---------------------------------------------------------------------------
// Phase 2: benchmark.
// ---------------------------------------------------------------------------

/// Number of warmup iterations before the timed run. Warmup lets the CPU
/// settle (branch predictor, caches, frequency scaling) so the first call's
/// outlier doesn't skew the result.
const WARMUP: usize = 10;
/// Number of timed iterations. Higher = more stable median, longer wall time.
/// 200 is enough that even the slowest parameter set finishes in seconds.
const ITERS: usize = 200;

#[derive(Clone, Copy)]
struct Stats {
    min: Duration,
    median: Duration,
    mean: Duration,
    max: Duration,
}

fn summarize(mut samples: Vec<Duration>) -> Stats {
    samples.sort_unstable();
    let n = samples.len();
    let sum: Duration = samples.iter().sum();
    Stats {
        min: samples[0],
        median: samples[n / 2],
        mean: sum / n as u32,
        max: samples[n - 1],
    }
}

fn fmt_dur(d: Duration) -> String {
    let us = d.as_secs_f64() * 1e6;
    if us < 1.0 {
        format!("{:7.0} ns", us * 1000.0)
    } else if us < 1000.0 {
        format!("{:7.2} us", us)
    } else {
        format!("{:7.2} ms", us / 1000.0)
    }
}

fn ops_per_sec(median: Duration) -> f64 {
    1.0 / median.as_secs_f64()
}

fn print_stats(name: &str, s: Stats) {
    println!(
        "    {name:<8}  min {}  median {}  mean {}  max {}  ({:>8.0} ops/s)",
        fmt_dur(s.min),
        fmt_dur(s.median),
        fmt_dur(s.mean),
        fmt_dur(s.max),
        ops_per_sec(s.median),
    );
}

fn bench<P: PkeParams>(label: &str, rng: &mut SplitMix64) {
    println!("  {label}");

    // Pre-generate a stable key pair, message, and randomness for the
    // Encrypt/Decrypt benchmarks. We use the SAME key across iterations
    // because that's what cycle-cost measurements normally do; mixing
    // KeyGen into Encrypt would conflate two very different operations.
    let d = rand_b32(rng);
    let (dk, ek): (DecryptionKey<P>, EncryptionKey<P>) = DecryptionKey::<P>::generate(&d);
    let msg = rand_b32(rng);
    let r = rand_b32(rng);
    let ct_fixed = ek.encrypt(&msg, &r);

    // -------- KeyGen --------
    // Generate a fresh seed each iteration so we're actually measuring the
    // work, not memoization.
    for _ in 0..WARMUP {
        let d = rand_b32(rng);
        let (dk, ek) = DecryptionKey::<P>::generate(&d);
        black_box(&dk);
        black_box(&ek);
    }
    let mut samples = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let d = rand_b32(rng);
        let t0 = Instant::now();
        let pair = DecryptionKey::<P>::generate(black_box(&d));
        samples.push(t0.elapsed());
        black_box(pair);
    }
    print_stats("KeyGen", summarize(samples));

    // -------- Encrypt --------
    // Same ek, fresh (msg, r) each iteration. msg/r are tiny (32B each), so
    // the cost of generating them is negligible compared to encrypt itself.
    for _ in 0..WARMUP {
        let m = rand_b32(rng);
        let rr = rand_b32(rng);
        let c = ek.encrypt(&m, &rr);
        black_box(c);
    }
    let mut samples = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let m = rand_b32(rng);
        let rr = rand_b32(rng);
        let t0 = Instant::now();
        let c = ek.encrypt(black_box(&m), black_box(&rr));
        samples.push(t0.elapsed());
        black_box(c);
    }
    print_stats("Encrypt", summarize(samples));

    // -------- Decrypt --------
    // Same dk, same ct (the cost of decrypt is independent of the
    // ciphertext's content; it just runs the same NTT + inverse NTT).
    for _ in 0..WARMUP {
        let m = dk.decrypt(&ct_fixed);
        black_box(m);
    }
    let mut samples = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t0 = Instant::now();
        let m = dk.decrypt(black_box(&ct_fixed));
        samples.push(t0.elapsed());
        black_box(m);
    }
    print_stats("Decrypt", summarize(samples));

    println!();
}

// ---------------------------------------------------------------------------

fn main() {
    let mut rng = SplitMix64(seed_from_time());

    println!("=== Phase 1: K-PKE round-trip (KeyGen -> Encrypt -> Decrypt) ===\n");
    round_trip::<MlKem512>("MlKem512", &mut rng);
    round_trip::<MlKem768>("MlKem768", &mut rng);
    round_trip::<MlKem1024>("MlKem1024", &mut rng);

    println!("\n=== Phase 2: benchmark ({ITERS} timed iterations, {WARMUP} warmup) ===");
    if cfg!(debug_assertions) {
        println!();
        println!("  !! WARNING: this is a debug build. Numbers are not meaningful.");
        println!("  !! Re-run with: cargo run --release --bin pke_demo");
    }
    println!();
    bench::<MlKem512>("MlKem512", &mut rng);
    bench::<MlKem768>("MlKem768", &mut rng);
    bench::<MlKem1024>("MlKem1024", &mut rng);
}

fn hex_prefix(b: &[u8], n: usize) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    for x in &b[..n.min(b.len())] {
        let _ = write!(s, "{:02x}", x);
    }
    s
}