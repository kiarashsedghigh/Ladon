//! Cost estimator for **Micciancio & Suhl**, "Simulation-Secure Threshold PKE
//! from LWE with Polynomial Modulus", IACR Communications in Cryptology Vol 1
//! No 4 (2024), https://doi.org/10.62056/a0zogy4e.
//!
//! Sweeps N ∈ {4, 8, 16, 32} parties at the Section-6 parameter set
//! (modelled on Frodo-640):
//!   n = 640,   q = 65537 (~2^16),
//!   sigma_dLWE = 5  (for s, e_pk),
//!   sigma_e   ≈ 11.09  (for r, f during encryption),
//!   sigma_sm  ≈ 7.07   (smudging noise during partial decryption).
//!
//! Supports up to T ≈ 8263 parties before the noise budget runs out.
//!
//! Why this is fundamentally cheaper than the polynomial-ring schemes
//! ------------------------------------------------------------------
//! Plain LWE, NOT Ring-LWE. There are no polynomials, no NTT, no Montgomery
//! tricks. Every operation is dense matrix-vector or vector-vector mod a
//! 17-bit prime, which fits in plain u32 with u64 accumulators. Specifically:
//!
//!   * KeyGen:        1 matrix-vector mult (A·s, n² = 409 600 u32×u32) +
//!                    Gaussians (2n samples at σ ≈ 5) +
//!                    additive (T,T) sharing of s (n·T scalar samples).
//!   * Encrypt:       1 matrix-vector mult (r^T A) + 1 inner product (r^T b) +
//!                    Gaussians (2n samples at σ ≈ 11) + 1 continuous Gaussian.
//!   * Partial Dec:   1 INNER PRODUCT only (n = 640 u32 mults) +
//!                    1 smudging-noise Gaussian sample.
//!                    LOCAL: no interaction between parties, 0 communication rounds.
//!                    Output: ONE u32 scalar (~17 bits, 3 bytes wire).
//!   * Combine:       (T−1) scalar additions + 1 subtraction + 1 decode.
//!                    Aggregator collects T scalars (3·T bytes), sums, decodes.
//!
//! Contrast with the polynomial-ring schemes already benchmarked:
//!
//!   scheme               cost per phase            comm/round
//!   ------------------   -----------------------   ---------------------------
//!   sasha1 (q~2^100)     12-18 RNS poly muls       3 rounds, full polys on wire
//!   sasha2 (q~2^50)      3-5 NTT poly muls         3 rounds, full polys
//!   Micciancio-Suhl      1 vector inner product    0 rounds, single scalar
//!
//! The cost gap is ~3-4 orders of magnitude in the threshold-decryption phase.
//!
//! Build instructions
//! ------------------
//!   Drop this file into examples/bench_micciancio_suhl.rs and run:
//!     cargo run --release --example bench_micciancio_suhl

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::time::{Duration, Instant};

// =========================================================================
// Section-6 parameters (modelled on Frodo-640).
// =========================================================================

const N: usize         = 640;
const Q: u32           = 65537;
const Q64: u64         = 65537;
const SIGMA_DLWE: f64  = 5.0;
const SIGMA_E: f64     = 11.09;
const SIGMA_SM: f64    = 7.07;

/// Approximate upper bound on √c where c = ‖s‖² + ‖e_pk‖² (from paper §6).
/// Used as the scale factor for the continuous Gaussian e' in Encrypt.
const SQRT_C_BOUND: f64 = 91.053;

const PARTY_COUNTS: &[usize] = &[4, 8, 16, 32];

// =========================================================================
// Scalar arithmetic mod Q (q = 65537).
// =========================================================================

#[inline(always)] fn redq(x: u64) -> u32 { (x % Q64) as u32 }

#[inline(always)] fn add_mod_q(a: u32, b: u32) -> u32 {
    let s = a + b; if s >= Q { s - Q } else { s }
}

#[inline(always)] fn sub_mod_q(a: u32, b: u32) -> u32 {
    if a >= b { a - b } else { a + Q - b }
}

// =========================================================================
// Linear-algebra primitives.
//
// At n = 640 and q < 2^17, every product fits in u32 and the row-accumulator
// for a matrix-vector mult is bounded by n·(q-1)² ≈ 2^41.3, well within u64.
// We reduce mod q ONCE per output coordinate. No NTT, no Montgomery.
// =========================================================================

type Vec_n = Vec<u32>;
type Mat_n = Vec<u32>; // row-major, length n*n

/// y := A · x mod q  (rows × cols, x has length cols, y has length rows).
fn matvec(a: &[u32], x: &[u32], rows: usize, cols: usize) -> Vec<u32> {
    let mut y = vec![0u32; rows];
    for i in 0..rows {
        let row = &a[i * cols..(i + 1) * cols];
        let mut acc: u64 = 0;
        for j in 0..cols {
            acc += row[j] as u64 * x[j] as u64;
        }
        y[i] = redq(acc);
    }
    y
}

/// y := A^T · r mod q  (rows × cols, r has length rows, y has length cols).
/// Iterates A in row-major order for cache locality even though we want A^T·r.
fn matvec_transposed(a: &[u32], r: &[u32], rows: usize, cols: usize) -> Vec<u32> {
    let mut y = vec![0u64; cols];
    for i in 0..rows {
        let ri = r[i] as u64;
        let row = &a[i * cols..(i + 1) * cols];
        for j in 0..cols {
            y[j] += ri * row[j] as u64;
        }
    }
    y.into_iter().map(redq).collect()
}

fn inner_product(a: &[u32], b: &[u32]) -> u32 {
    let mut acc: u64 = 0;
    for i in 0..a.len() {
        acc += a[i] as u64 * b[i] as u64;
    }
    redq(acc)
}

fn vec_add_mod(a: &[u32], b: &[u32]) -> Vec<u32> {
    a.iter().zip(b.iter()).map(|(&x, &y)| add_mod_q(x, y)).collect()
}

// =========================================================================
// Sampling.
// =========================================================================

fn sample_uniform_vec(len: usize, rng: &mut StdRng) -> Vec<u32> {
    (0..len).map(|_| rng.gen_range(0..Q)).collect()
}

fn sample_uniform_matrix(rows: usize, cols: usize, rng: &mut StdRng) -> Vec<u32> {
    (0..rows * cols).map(|_| rng.gen_range(0..Q)).collect()
}

/// Box-Muller pair: two i32 samples per (ln, sqrt, cos, sin) call.
fn box_muller_pair(sigma: f64, rng: &mut StdRng) -> (i32, i32) {
    let u1: f64 = rng.gen_range(1e-12_f64..1.0);
    let u2: f64 = rng.gen_range(0.0_f64..1.0);
    let r = (-2.0 * u1.ln()).sqrt() * sigma;
    let theta = 2.0 * std::f64::consts::PI * u2;
    ((r * theta.cos()).round() as i32, (r * theta.sin()).round() as i32)
}

/// Discrete-Gaussian (rounded continuous) vector mod q.
fn sample_gauss_vec(sigma: f64, len: usize, rng: &mut StdRng) -> Vec<u32> {
    let qi = Q as i32;
    let mut out = vec![0u32; len];
    let mut i = 0;
    while i < len {
        let (g1, g2) = box_muller_pair(sigma, rng);
        out[i] = g1.rem_euclid(qi) as u32;
        if i + 1 < len {
            out[i + 1] = g2.rem_euclid(qi) as u32;
            i += 2;
        } else {
            i += 1;
        }
    }
    out
}

fn sample_gauss_scalar(sigma: f64, rng: &mut StdRng) -> u32 {
    let (g1, _) = box_muller_pair(sigma, rng);
    g1.rem_euclid(Q as i32) as u32
}

// =========================================================================
// Threshold PKE phases. Construction is the LP11-like scheme of §5.1.
// =========================================================================

struct PublicKey { a: Mat_n, b: Vec_n }

/// (T,T) additive sharing of s: s_1, ..., s_{T-1} uniform, s_T := s - sum.
fn additive_share(s: &Vec_n, t_parties: usize, rng: &mut StdRng) -> Vec<Vec_n> {
    let mut shares: Vec<Vec_n> = (0..t_parties - 1)
        .map(|_| sample_uniform_vec(N, rng))
        .collect();
    let mut last = s.clone();
    for sh in &shares {
        for i in 0..N {
            last[i] = sub_mod_q(last[i], sh[i]);
        }
    }
    shares.push(last);
    shares
}

/// KeyGen: 1 matvec (A·s) + 2 Gaussian vectors + additive sharing.
fn keygen(t_parties: usize, rng: &mut StdRng) -> (PublicKey, Vec<Vec_n>) {
    let a    = sample_uniform_matrix(N, N, rng);
    let s    = sample_gauss_vec(SIGMA_DLWE, N, rng);
    let e_pk = sample_gauss_vec(SIGMA_DLWE, N, rng);
    let as_  = matvec(&a, &s, N, N);
    let b    = vec_add_mod(&as_, &e_pk);
    let shares = additive_share(&s, t_parties, rng);
    (PublicKey { a, b }, shares)
}

/// Encrypt: ciphertext = (r^T A + f, r^T b + e' + msg).
/// 1 matvec (r^T A) + 1 inner product (r^T b) + 2n Gaussians + 1 cont. Gaussian.
fn encrypt(pk: &PublicKey, rng: &mut StdRng) -> (Vec_n, u32) {
    let r = sample_gauss_vec(SIGMA_E, N, rng);
    let f = sample_gauss_vec(SIGMA_E, N, rng);
    let r_t_a   = matvec_transposed(&pk.a, &r, N, N);
    let a_prime = vec_add_mod(&r_t_a, &f);
    let rt_b    = inner_product(&r, &pk.b);
    // e' has width sigma_e * sqrt(c) ≈ 11.09 · 91.05 ≈ 1010.
    let e_prime = sample_gauss_scalar(SIGMA_E * SQRT_C_BOUND, rng);
    let msg     = 0u32; // for benchmarking; encoding is msg·⌊q/p⌋
    let b_prime = add_mod_q(add_mod_q(rt_b, e_prime), msg);
    (a_prime, b_prime)
}

/// Partial decryption: 0-round, fully local.
/// Cost = 1 inner product + 1 smudging Gaussian. Outputs ONE u32 scalar.
fn partial_decrypt(a_prime: &[u32], s_i: &[u32], rng: &mut StdRng) -> u32 {
    let dot     = inner_product(a_prime, s_i);
    let e_tilde = sample_gauss_scalar(SIGMA_SM, rng);
    add_mod_q(dot, e_tilde)
}

/// Combine: aggregator sums T scalars, subtracts from b', decodes.
fn combine(b_prime: u32, partials: &[u32]) -> u32 {
    let mut sum: u32 = 0;
    for &d in partials { sum = add_mod_q(sum, d); }
    let masked = sub_mod_q(b_prime, sum);
    // Decode 2-bit msg by rounding mod q to nearest multiple of q/4.
    // Use centred representative in (-q/2, q/2].
    let x = if masked > Q / 2 { masked as i64 - Q as i64 } else { masked as i64 };
    let q4 = (Q / 4) as i64;
    let rounded = if x >= 0 {
        ((x + q4 / 2) / q4) % 4
    } else {
        (-(((-x) + q4 / 2) / q4)).rem_euclid(4)
    };
    rounded as u32
}

// =========================================================================
// Driver
// =========================================================================

#[derive(Clone, Copy)]
struct Timing {
    setup: Duration, encaps: Duration, share: Duration, combine: Duration,
}

fn run_one(t_parties: usize, rng: &mut StdRng) -> Timing {
    let t0 = Instant::now();
    let (pk, shares) = keygen(t_parties, rng);
    let setup = t0.elapsed();

    let t0 = Instant::now();
    let (a_prime, b_prime) = encrypt(&pk, rng);
    let encaps = t0.elapsed();

    // Time ONE party's partial decryption (the others are independent / parallel).
    let t0 = Instant::now();
    std::hint::black_box(partial_decrypt(&a_prime, &shares[0], rng));
    let share = t0.elapsed();

    // Then compute all T to feed the aggregator.
    let partials: Vec<u32> = shares.iter()
        .map(|s_i| partial_decrypt(&a_prime, s_i, rng))
        .collect();

    let t0 = Instant::now();
    std::hint::black_box(combine(b_prime, &partials));
    let combine = t0.elapsed();

    Timing { setup, encaps, share, combine }
}

fn fmt_d(d: Duration) -> String {
    let ns = d.as_nanos() as f64;
    if ns >= 1_000_000_000.0 { format!("{:>9.2} s",  ns / 1_000_000_000.0) }
    else if ns >= 1_000_000.0 { format!("{:>9.2} ms", ns / 1_000_000.0) }
    else if ns >= 1000.0     { format!("{:>9.2} us", ns / 1000.0) }
    else                     { format!("{:>9.2} ns", ns) }
}

fn main() {
    println!("===== Micciancio-Suhl 2024 cost estimate =====");
    println!("Simulation-Secure Threshold PKE from LWE with Polynomial Modulus");
    println!();
    println!("Parameters (paper §6, modelled on Frodo-640):");
    println!("  n          = {}", N);
    println!("  q          = {} (~2^{:.2})", Q, (Q as f64).log2());
    println!("  sigma_dLWE = {} (s, e_pk)", SIGMA_DLWE);
    println!("  sigma_e    = {:.2} (r, f)", SIGMA_E);
    println!("  sigma_sm   = {:.2} (smudging in partial dec)", SIGMA_SM);
    println!("Mode: plain LWE (no ring), T-of-T additive sharing, 0-round local dec");
    println!();
    println!("Wire sizes:");
    println!("  |pk|         ≈ {:>10} bytes (A: {} bytes, b: {} bytes)",
             N * N * 2 + N * 2 + 4, N * N * 2, N * 2);
    println!("                  (~{:.1} KiB — A can be regenerated from a 32-byte seed à la Frodo)",
             (N * N * 2 + N * 2 + 4) as f64 / 1024.0);
    println!("  |ct|         ≈ {:>10} bytes ((a', b'): {} + 2)", N * 2 + 2, N * 2);
    println!("  |partial dec|≈ {:>10} bytes  <-- 3 bytes per party per ciphertext!",
             2);
    println!();

    let mut rng = StdRng::from_entropy();

    // Per-op calibration.
    let a = sample_uniform_matrix(N, N, &mut rng);
    let x = sample_uniform_vec(N, &mut rng);
    let r = sample_uniform_vec(N, &mut rng);

    const N_CAL_MV: u32   = 200;
    const N_CAL_IP: u32   = 100_000;
    const N_CAL_GA: u32   = 100_000;

    let t0 = Instant::now();
    for _ in 0..N_CAL_MV { std::hint::black_box(matvec(&a, &x, N, N)); }
    let mv = t0.elapsed() / N_CAL_MV;

    let t0 = Instant::now();
    for _ in 0..N_CAL_MV { std::hint::black_box(matvec_transposed(&a, &r, N, N)); }
    let mv_t = t0.elapsed() / N_CAL_MV;

    let t0 = Instant::now();
    for _ in 0..N_CAL_IP { std::hint::black_box(inner_product(&x, &r)); }
    let ip = t0.elapsed() / N_CAL_IP;

    let t0 = Instant::now();
    for _ in 0..N_CAL_GA { std::hint::black_box(sample_gauss_scalar(SIGMA_SM, &mut rng)); }
    let gs = t0.elapsed() / N_CAL_GA;

    println!("Per-op calibration:");
    println!("  matrix-vector mult  A·x    (n×n) : {}", fmt_d(mv));
    println!("  matrix-vector mult  A^T·r  (n×n) : {}", fmt_d(mv_t));
    println!("  vector inner product       (n)   : {}", fmt_d(ip));
    println!("  one Gaussian sample (Box-Muller) : {}", fmt_d(gs));
    println!();

    // Sweep.
    println!("====================================================================================");
    println!("              Per-phase timings (plain LWE matrix-vector, single-thread)");
    println!("====================================================================================");
    println!("{:>3} {:>11} {:>11} {:>11} {:>11} {:>12} {:>12}",
             "N", "Setup", "Encaps", "Share/p", "Combine", "Decap(par)", "Decap(seq)");
    println!("{}", "-".repeat(82));

    for &t in PARTY_COUNTS {
        // Warm up once so the matrix is in L2/L3 before timing.
        let _warm = run_one(t, &mut rng);
        let tm = run_one(t, &mut rng);
        let dec_par = tm.share + tm.combine;
        let dec_seq = tm.share * (t as u32) + tm.combine;
        println!("{:>3} {:>11} {:>11} {:>11} {:>11} {:>12} {:>12}",
                 t,
                 fmt_d(tm.setup), fmt_d(tm.encaps),
                 fmt_d(tm.share), fmt_d(tm.combine),
                 fmt_d(dec_par), fmt_d(dec_seq));
    }

    println!();
    println!("Legend:");
    println!("  Share/p     = ONE party's partial decryption (LOCAL, 0-round)");
    println!("                  cost = 1 inner product + 1 Gaussian sample");
    println!("                  output = 1 scalar (~3 bytes wire)");
    println!("  Combine     = aggregator: sum T scalars, subtract, decode");
    println!("  Decap(par)  = Share/p + Combine  (parties run in parallel; standard mode)");
    println!("  Decap(seq)  = N · Share/p + Combine  (single-machine simulation)");
    println!();
    println!("Notes:");
    println!("  * No NTT, no Montgomery, no polynomial ring. Just dense u32 mat-vec.");
    println!("  * Partial decryption is 0-round and embarrassingly local — no party-to-party");
    println!("    communication, just a single u32 scalar sent to the aggregator.");
    println!("  * Each ciphertext should be partially-decrypted AT MOST ONCE per party");
    println!("    (Theorem 1, third bullet): repeated calls with the same a would let the");
    println!("    adversary average out the smudging noise.");
    println!("  * Partial decryption output is ONE scalar mod q ≈ 17 bits.  Compare to");
    println!("    the polynomial-ring schemes' ShareExtract output: up to several KiB.");
    println!("  * Public key A (~800 KiB) is normally regenerated from a 32-byte seed");
    println!("    Frodo-style; wire size is then dominated by b (~1.3 KiB).");
}