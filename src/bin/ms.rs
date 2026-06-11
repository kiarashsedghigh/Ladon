//! Cost estimator for **Micciancio-Suhl with Shamir sharing** — i.e., the
//! same plain-LWE / linear-algebra construction as MS25, but with the
//! K-of-K additive sharing replaced by t-of-K Shamir over Z_q.
//!
//! Why this file exists
//! --------------------
//! The MS25 paper only supports K-of-K threshold (additive sharing).
//! When comparing its cost to t-of-K schemes (Pilvi, sasha1, sasha2),
//! the additive KeyGen line is unfair: additive sharing is essentially
//! free (K-1 random vectors), while t-of-K Shamir requires a degree-
//! (t-1) polynomial evaluation at K distinct points per secret coordinate.
//! This estimator answers: "if we kept the MS25 plain-LWE structure but
//! used Shamir sharing for true t-of-K threshold, what would KeyGen and
//! Combine cost?"
//!
//! ParDec is unchanged — still one inner product + one Gaussian, output
//! = one Z_q scalar. What changes:
//!
//!   ADDITIVE (original MS25)             SHAMIR (this file)
//!   -----------------------------        --------------------------------
//!   KeyGen share: ~N*(K-1) samples       KeyGen share: N*(t-1) random
//!     (cost ~ memory copy)                 coeffs + N*K*(t-1) mod-q mults
//!                                          (Horner over each coord)
//!   Combine: sum K partials              Combine: 1 mod-q inverse (Fermat)
//!     (T-1 add_mod_q)                      + t*(t-1)/2 mults for Lagrange
//!                                          coeffs + t Lagrange-weighted
//!                                          partials.
//!
//! Notes on correctness (not addressed here)
//! -----------------------------------------
//! Reconstruction multiplies each ParDec partial (which contains smudging
//! noise) by a Lagrange coefficient λ_j whose norm in the worst case
//! grows with K choose t — exactly the issue Pilvi solves via subtractive
//! sets. For the Shamir version of MS to be cryptographically sound at
//! q = 65537, the smudging-noise budget needs to absorb those Lagrange
//! coefficients, which it won't for non-trivial K, t. So:
//!
//!   * Numerically, q = 65537 (Frodo-640) DOES NOT WORK with Shamir
//!     reconstruction for K, t > a handful.  The cost numbers below are
//!     still meaningful as a structural comparison — they tell you what
//!     the LINEAR-ALGEBRA cost of going t-of-K with naive Shamir looks
//!     like — but the parameter set is not the one you'd actually pick.
//!   * A real Shamir-MS construction would need noise flooding
//!     (super-poly q) or a low-norm sharing scheme (subtractive sets,
//!     {0,1}-LSSS).
//!
//! Sweep:
//!   (K, t) in { (8, 2), (8, 6), (16, 10), (32, 16) }
//! mirroring Pilvi's Table 5 to allow side-by-side comparison.
//!
//! Build instructions
//! ------------------
//!   Drop into examples/bench_micciancio_suhl_shamir.rs and run:
//!     cargo run --release --example bench_micciancio_suhl_shamir

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::time::{Duration, Instant};

// =========================================================================
// Base parameters (Frodo-640, unchanged from MS25 §6).
// =========================================================================

const N: usize         = 640;
const Q: u32           = 65537;
const Q64: u64         = 65537;
const SIGMA_DLWE: f64  = 5.0;
const SIGMA_E: f64     = 11.09;
const SIGMA_SM: f64    = 7.07;
const SQRT_C_BOUND: f64 = 91.053;

#[derive(Clone, Copy)]
struct ParamSet {
    name: &'static str,
    k_parties: usize,
    t_thresh:  usize,
}

// Mirroring Pilvi Table 5's (t, K) selection.
const PARAMS: &[ParamSet] = &[
    ParamSet { name: "(K= 8, t= 2)", k_parties:  8, t_thresh:  2 },
    ParamSet { name: "(K= 8, t= 6)", k_parties:  8, t_thresh:  6 },
    ParamSet { name: "(K=16, t=10)", k_parties: 16, t_thresh: 10 },
    ParamSet { name: "(K=32, t=16)", k_parties: 32, t_thresh: 16 },
];

// =========================================================================
// Scalar arithmetic mod Q (q = 65537), plus modular inverse for Lagrange.
// =========================================================================

#[inline(always)] fn redq(x: u64) -> u32 { (x % Q64) as u32 }
#[inline(always)] fn add_mod_q(a: u32, b: u32) -> u32 {
    let s = a + b; if s >= Q { s - Q } else { s }
}
#[inline(always)] fn sub_mod_q(a: u32, b: u32) -> u32 {
    if a >= b { a - b } else { a + Q - b }
}
#[inline(always)] fn mul_mod_q(a: u32, b: u32) -> u32 {
    ((a as u64 * b as u64) % Q64) as u32
}

/// Modular exponentiation a^e mod Q.
fn pow_mod_q(mut a: u64, mut e: u64) -> u64 {
    let mut r: u64 = 1;
    a %= Q64;
    while e > 0 {
        if e & 1 == 1 { r = (r * a) % Q64; }
        a = (a * a) % Q64;
        e >>= 1;
    }
    r
}

/// Modular inverse a^{-1} mod Q via Fermat (Q prime).
fn inv_mod_q(a: u32) -> u32 { pow_mod_q(a as u64, Q64 - 2) as u32 }

// =========================================================================
// Linear-algebra primitives (identical to original MS estimator).
// =========================================================================

fn matvec(a: &[u32], x: &[u32], rows: usize, cols: usize) -> Vec<u32> {
    let mut y = vec![0u32; rows];
    for i in 0..rows {
        let row = &a[i * cols..(i + 1) * cols];
        let mut acc: u64 = 0;
        for j in 0..cols { acc += row[j] as u64 * x[j] as u64; }
        y[i] = redq(acc);
    }
    y
}

fn matvec_transposed(a: &[u32], r: &[u32], rows: usize, cols: usize) -> Vec<u32> {
    let mut y = vec![0u64; cols];
    for i in 0..rows {
        let ri = r[i] as u64;
        let row = &a[i * cols..(i + 1) * cols];
        for j in 0..cols { y[j] += ri * row[j] as u64; }
    }
    y.into_iter().map(redq).collect()
}

fn inner_product(a: &[u32], b: &[u32]) -> u32 {
    let mut acc: u64 = 0;
    for i in 0..a.len() { acc += a[i] as u64 * b[i] as u64; }
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

fn box_muller_pair(sigma: f64, rng: &mut StdRng) -> (i32, i32) {
    let u1: f64 = rng.gen_range(1e-12_f64..1.0);
    let u2: f64 = rng.gen_range(0.0_f64..1.0);
    let r = (-2.0 * u1.ln()).sqrt() * sigma;
    let theta = 2.0 * std::f64::consts::PI * u2;
    ((r * theta.cos()).round() as i32, (r * theta.sin()).round() as i32)
}
fn sample_gauss_vec(sigma: f64, len: usize, rng: &mut StdRng) -> Vec<u32> {
    let qi = Q as i32;
    let mut out = vec![0u32; len];
    let mut i = 0;
    while i < len {
        let (g1, g2) = box_muller_pair(sigma, rng);
        out[i] = g1.rem_euclid(qi) as u32;
        if i + 1 < len { out[i + 1] = g2.rem_euclid(qi) as u32; i += 2; } else { i += 1; }
    }
    out
}
fn sample_gauss_scalar(sigma: f64, rng: &mut StdRng) -> u32 {
    let (g1, _) = box_muller_pair(sigma, rng);
    g1.rem_euclid(Q as i32) as u32
}

// =========================================================================
// Shamir sharing over Z_q.
//
// For each coordinate i in [N]:
//   f_i(X) = s[i] + c_{i,1} X + ... + c_{i,t-1} X^{t-1}   (random c_{i,j})
//   share_k[i] = f_i(alpha_k)   for alpha_k = k+1, k = 0..K-1
//
// Cost per coordinate: (t-1) random samples + K * (t-1) Horner mults.
// Total: N * (t-1) samples + N * K * (t-1) mults.
//
// Evaluation points alpha_k = 1..K  (q = 65537, K <= 32  far below q).
// =========================================================================

fn shamir_share(s: &[u32], k_parties: usize, t_thresh: usize,
                rng: &mut StdRng) -> Vec<Vec<u32>>
{
    let n = s.len();
    // coeffs[j][i] = coefficient of X^j for coordinate i. coeffs[0] = s.
    let mut coeffs: Vec<Vec<u32>> = vec![vec![0u32; n]; t_thresh];
    coeffs[0] = s.to_vec();
    for j in 1..t_thresh {
        for i in 0..n { coeffs[j][i] = rng.gen_range(0..Q); }
    }
    // Evaluate at alpha_k = k+1 via Horner.
    let mut shares = vec![vec![0u32; n]; k_parties];
    for k in 0..k_parties {
        let alpha = (k + 1) as u64;
        for i in 0..n {
            let mut acc: u64 = coeffs[t_thresh - 1][i] as u64;
            for j in (0..t_thresh - 1).rev() {
                acc = (acc * alpha + coeffs[j][i] as u64) % Q64;
            }
            shares[k][i] = acc as u32;
        }
    }
    shares
}

/// Lagrange coefficients λ_j evaluated at 0, for a subset T = {alpha_j}_{j in [t]}.
/// λ_j = ∏_{k ≠ j} (-alpha_k) / (alpha_j - alpha_k)  mod q.
///
/// One inverse per j (or one inverse total via batched-Montgomery; we do the
/// naive one-per-j here since t is at most ~16). Cost: t * (t-1) mults +
/// t modular inverses (each ~ 17 squarings via Fermat). Computed ONCE per
/// Combine call.
fn lagrange_at_zero(t_set: &[u32]) -> Vec<u32> {
    let t = t_set.len();
    let mut lambda = vec![0u32; t];
    for j in 0..t {
        let aj = t_set[j] as u64;
        let mut num: u64 = 1;
        let mut den: u64 = 1;
        for k in 0..t {
            if k == j { continue; }
            let ak = t_set[k] as u64;
            num = (num * ((Q64 - ak) % Q64)) % Q64;            // (-alpha_k) mod q
            let diff = if aj >= ak { aj - ak } else { aj + Q64 - ak };
            den = (den * diff) % Q64;
        }
        let den_inv = inv_mod_q(den as u32);
        lambda[j] = ((num * den_inv as u64) % Q64) as u32;
    }
    lambda
}

// =========================================================================
// Threshold PKE phases.
// =========================================================================

struct PublicKey { a: Vec<u32>, b: Vec<u32> }

/// KeyGen with Shamir t-of-K sharing.
///   1 matvec (A·s)  +  2 Gaussian vectors  +  Shamir share of s.
fn keygen(ps: &ParamSet, rng: &mut StdRng) -> (PublicKey, Vec<Vec<u32>>) {
    let a    = sample_uniform_matrix(N, N, rng);
    let s    = sample_gauss_vec(SIGMA_DLWE, N, rng);
    let e_pk = sample_gauss_vec(SIGMA_DLWE, N, rng);
    let as_  = matvec(&a, &s, N, N);
    let b    = vec_add_mod(&as_, &e_pk);
    let shares = shamir_share(&s, ps.k_parties, ps.t_thresh, rng);
    (PublicKey { a, b }, shares)
}

/// Encrypt: unchanged from additive-MS.
fn encrypt(pk: &PublicKey, rng: &mut StdRng) -> (Vec<u32>, u32) {
    let r = sample_gauss_vec(SIGMA_E, N, rng);
    let f = sample_gauss_vec(SIGMA_E, N, rng);
    let r_t_a = matvec_transposed(&pk.a, &r, N, N);
    let a_prime = vec_add_mod(&r_t_a, &f);
    let rt_b  = inner_product(&r, &pk.b);
    let e_prime = sample_gauss_scalar(SIGMA_E * SQRT_C_BOUND, rng);
    let b_prime = add_mod_q(add_mod_q(rt_b, e_prime), 0);
    (a_prime, b_prime)
}

/// Partial decryption: unchanged from additive-MS.
/// One inner product + one smudging Gaussian. Output: one Z_q scalar.
fn partial_decrypt(a_prime: &[u32], s_i: &[u32], rng: &mut StdRng) -> u32 {
    let dot = inner_product(a_prime, s_i);
    let e_tilde = sample_gauss_scalar(SIGMA_SM, rng);
    add_mod_q(dot, e_tilde)
}

/// Shamir combine: aggregator gets t partials from parties at indices T,
/// computes Lagrange coefficients at 0, then  Σ λ_j pd_j  mod q.
/// Cost: 1 Lagrange precompute (t*(t-1) mults + t inverses) + t mults.
fn combine_shamir(b_prime: u32, partials_at: &[(u32, u32)] /* (alpha, pd) */) -> u32 {
    let alphas: Vec<u32> = partials_at.iter().map(|&(a, _)| a).collect();
    let lambda = lagrange_at_zero(&alphas);
    let mut sum: u32 = 0;
    for j in 0..partials_at.len() {
        let prod = mul_mod_q(lambda[j], partials_at[j].1);
        sum = add_mod_q(sum, prod);
    }
    let masked = sub_mod_q(b_prime, sum);
    // Decode (unchanged from MS).
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
    setup: Duration, encaps: Duration, share: Duration,
    combine: Duration, sharing_only: Duration,
}

fn run_one(ps: &ParamSet, rng: &mut StdRng) -> Timing {
    // Full KeyGen including Shamir share.
    let t0 = Instant::now();
    let (pk, shares) = keygen(ps, rng);
    let setup = t0.elapsed();

    // Sharing-only sub-time: re-do just the Shamir step for breakdown.
    // (Gives us an apples-to-apples comparison with additive sharing later.)
    let s_tmp = sample_gauss_vec(SIGMA_DLWE, N, rng);
    let t0 = Instant::now();
    std::hint::black_box(shamir_share(&s_tmp, ps.k_parties, ps.t_thresh, rng));
    let sharing_only = t0.elapsed();

    let t0 = Instant::now();
    let (a_prime, b_prime) = encrypt(&pk, rng);
    let encaps = t0.elapsed();

    // One party's partial decryption (parties act in parallel, so this is
    // the wall-clock cost of the share phase regardless of t).
    let t0 = Instant::now();
    std::hint::black_box(partial_decrypt(&a_prime, &shares[0], rng));
    let share = t0.elapsed();

    // Build t partials from parties 1..t (alpha = 1..t).
    let partials_at: Vec<(u32, u32)> = (0..ps.t_thresh)
        .map(|k| ((k + 1) as u32, partial_decrypt(&a_prime, &shares[k], rng)))
        .collect();

    let t0 = Instant::now();
    std::hint::black_box(combine_shamir(b_prime, &partials_at));
    let combine = t0.elapsed();

    Timing { setup, encaps, share, combine, sharing_only }
}

fn fmt_d(d: Duration) -> String {
    let ns = d.as_nanos() as f64;
    if ns >= 1_000_000_000.0 { format!("{:>9.2} s",  ns / 1_000_000_000.0) }
    else if ns >= 1_000_000.0 { format!("{:>9.2} ms", ns / 1_000_000.0) }
    else if ns >= 1000.0     { format!("{:>9.2} us", ns / 1000.0) }
    else                     { format!("{:>9.2} ns", ns) }
}

fn main() {
    println!("===== Micciancio-Suhl with Shamir sharing =====");
    println!("Plain-LWE linear algebra, but with t-of-K Shamir over Z_q");
    println!("(replaces the K-of-K additive sharing in the original MS25).");
    println!();
    println!("Base parameters (paper §6, modelled on Frodo-640):");
    println!("  n = {}, q = {} (~2^{:.2})", N, Q, (Q as f64).log2());
    println!("  sigma_dLWE = {}, sigma_e = {:.2}, sigma_sm = {:.2}",
             SIGMA_DLWE, SIGMA_E, SIGMA_SM);
    println!();
    println!("WARNING: q = 65537 IS NOT a sound choice when Shamir reconstruction");
    println!("multiplies smudging noise by Lagrange coefficients. The numbers");
    println!("below model the LINEAR-ALGEBRA cost of going t-of-K with naive");
    println!("Shamir; a sound construction needs noise flooding or a low-norm");
    println!("sharing scheme (subtractive sets, as in Pilvi).");
    println!();

    let mut rng = StdRng::from_entropy();

    // Per-op calibration: cost of one Shamir share for various (K, t).
    println!("Shamir-share calibration (one N-coord share at varying K, t):");
    for ps in PARAMS {
        let s_tmp = sample_gauss_vec(SIGMA_DLWE, N, &mut rng);
        const N_CAL: u32 = 10;
        let t0 = Instant::now();
        for _ in 0..N_CAL {
            std::hint::black_box(shamir_share(&s_tmp, ps.k_parties, ps.t_thresh, &mut rng));
        }
        let avg = t0.elapsed() / N_CAL;
        println!("  {:<14}  N*K*(t-1) = {:>7} mults : {}", ps.name,
                 N * ps.k_parties * (ps.t_thresh - 1), fmt_d(avg));
    }
    println!();

    // Main sweep.
    println!("========================================================================================");
    println!("           Per-phase timings (plain LWE + Shamir t-of-K, single-thread)");
    println!("========================================================================================");
    println!("{:<14} {:>11} {:>11} {:>11} {:>11} {:>11} {:>12}",
             "config", "Setup", "  (share)", "Encaps", "PdShare", "Combine", "Decap(par)");
    println!("{}", "-".repeat(88));

    for ps in PARAMS {
        let _warm = run_one(ps, &mut rng);
        let tm = run_one(ps, &mut rng);
        let dec_par = tm.share + tm.combine;
        println!("{:<14} {:>11} {:>11} {:>11} {:>11} {:>11} {:>12}",
                 ps.name,
                 fmt_d(tm.setup), fmt_d(tm.sharing_only),
                 fmt_d(tm.encaps), fmt_d(tm.share),
                 fmt_d(tm.combine), fmt_d(dec_par));
    }
    println!();

    println!("Legend:");
    println!("  Setup       = full KeyGen (A*s + Gaussians + Shamir share)");
    println!("  (share)     = Shamir-share sub-cost only (for additive comparison)");
    println!("  PdShare     = ONE party's partial decryption (1 inner product +");
    println!("                  1 smudging Gaussian; output = 1 Z_q scalar)");
    println!("  Combine     = Lagrange-coeff precompute + t Lagrange-weighted partials");
    println!("  Decap(par)  = PdShare + Combine  (parties in parallel)");
    println!();

    println!("Op counts (Shamir vs additive):");
    println!("  KeyGen share:");
    println!("    additive: ~N*(K-1) random samples              (essentially memcpy)");
    println!("    Shamir:   N*(t-1) random samples +");
    println!("              N*K*(t-1) mod-q multiplications      (the new cost)");
    println!("  Combine:");
    println!("    additive: (K-1) add_mod_q                      (~tens of ns)");
    println!("    Shamir:   t Lagrange coeffs (t*(t-1) mults + t inverses)");
    println!("              + t mod-q multiply-adds              (~ a few us)");
    println!("  ParDec/party: UNCHANGED  (1 inner product + 1 Gaussian)");
    println!();

    println!("Takeaways:");
    println!("  * ParDec still ~ tens-of-us. The local 0-round property survives.");
    println!("  * Combine grows from O(K) adds to O(t^2) mults — still <100 us at t = 16.");
    println!("  * KeyGen's Shamir share scales as N*K*(t-1). At (K=32, t=16, N=640) that's");
    println!("    ~300k mod-q mults — a meaningful but still single-digit ms cost on top");
    println!("    of the existing A*s matvec (~400k mults).");
    println!("  * For a fair comparison against Pilvi: this number is the structural");
    println!("    floor for plain-LWE t-of-K KeyGen. Pilvi pays the *ring-mul* version of");
    println!("    this same cost (K*t*n ring mults via NTT), which is what its KGen line");
    println!("    in our Pilvi estimator measures.");
}