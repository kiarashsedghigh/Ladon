//! Cost estimator for **sasha2** = Boudgoust, del Pino, Lapiha, Prest,
//! "IND-CCA Lattice Threshold KEM under 30 KiB", PKC 2026 (ePrint 2026/021).
//!
//! Sweeps:
//!   * all 4 Table-1 parameter rows:
//!         (kappa, robust) in { (128, no), (128, yes), (256, no), (256, yes) }
//!   * all 4 party counts:  N in { 4, 8, 16, 32 }
//!
//! and prints per-phase + aggregate threshold-decap timings for each combo.
//!
//! Implementation notes
//! --------------------
//! Polynomial multiplication is a merged-psi negacyclic NTT (Pöppelmann-Oder-
//! Güneysu / Kyber-style), with the following layered optimisations:
//!
//!   * Montgomery multiplication: q ~ 2^50 is too large for a 32-bit modulus
//!     but fits comfortably in 64-bit Montgomery with R = 2^64. Every modular
//!     multiplication on the hot path is replaced by a u64 wrapping mul, a
//!     u128 fused mul-add, and a shift. The negative inverse -q^{-1} mod 2^64
//!     is computed once at startup by Newton/Hensel lifting (6 iterations).
//!   * All polynomial coefficients live in Montgomery form throughout the
//!     protocol. Conversion happens only at sampling boundaries (Gaussian
//!     and uniform). The inverse-NTT output is already in Montgomery form,
//!     so chained poly muls have zero conversion overhead.
//!   * Twiddle tables (psi^k, psi^{-k}) precomputed in Montgomery + bit-
//!     reversed index order. Indexed directly by the merged-psi NTT.
//!   * Gaussian sampler uses Box-Muller PAIRS (cos + sin from the same draw),
//!     halving the count of ln/sqrt and RNG calls.
//!   * Separate NTT contexts per ring degree built once at program start.
//!     d = 2048 (used by kappa = 128) and d = 4096 (used by kappa = 256).
//!   * Startup self-test: random poly mul via NTT vs schoolbook must agree
//!     for both d, before any timing is reported.
//!
//! Per-phase operation counts (taken straight from sasha2's algorithms):
//!   TIBE.Setup           Alg. 1   : 1 poly mul + 2 Gaussian polys + (T,N) share
//!   TIBE.Encrypt         Alg. 9   : 4 poly muls + 5 Gaussian polys + 1 uniform
//!   ShareExtract / party Algs.10-12: 5 poly muls + 4 Gaussian polys + 1 uniform
//!   Combine, NON-robust  Alg. 14  : 3 poly muls (final reconstruction)
//!   Combine, ROBUST      Alg. 14  : 3 N + 3 poly muls (N ShareVerify + final)
//!
//! Things this estimate does NOT include
//! -------------------------------------
//!   * BCHK+ signature (ML-DSA-44: ~50 us verify, ~500 us sign; Falcon-512
//!     similar) — present in TKEM.Encaps and Combine but cheap compared to
//!     the lattice operations at sasha2's parameters.
//!   * Random-oracle calls (Hcmt, Hid). SHA3/SHAKE on a few KiB is sub-100us.
//!   * Network. This is single-machine local-compute.
//!
//! Build instructions
//! ------------------
//!   Drop into examples/bench_sasha2.rs (or src/bin/), then:
//!     cargo run --release --example bench_sasha2

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::time::{Duration, Instant};

// =========================================================================
// Parameter sets: 4 rows from sasha2 Table 1.
// (sigma_p' = 2^27 across all rows; encoded once below.)
// =========================================================================

const SIGMA_P_PRIME: f64 = 134_217_728.0; // = 2^27

#[derive(Clone, Copy)]
struct ParamSet {
    name: &'static str,
    kappa: usize,
    robust: bool,
    d: usize,
    log_d: u32,
    sigma_s: f64,
    sigma_p: f64,
    sigma_r: f64,
    ek_bytes: usize,
    ct_bytes: usize,
}

const PS_128_NR: ParamSet = ParamSet {
    name: "k=128 non-robust",
    kappa: 128, robust: false,
    d: 2048, log_d: 11,
    sigma_s: 26.0,
    sigma_p: 34_359_738_368.0,  // 2^35
    sigma_r: 1.0,
    ek_bytes: 6688, ct_bytes: 28544,
};
const PS_128_R: ParamSet = ParamSet {
    name: "k=128 robust",
    kappa: 128, robust: true,
    d: 2048, log_d: 11,
    sigma_s: 25.0,
    sigma_p: 536_870_912.0,     // 2^29
    sigma_r: 1.0,
    ek_bytes: 8224, ct_bytes: 30368,
};
const PS_256_NR: ParamSet = ParamSet {
    name: "k=256 non-robust",
    kappa: 256, robust: false,
    d: 4096, log_d: 12,
    sigma_s: 26.0,
    sigma_p: 34_359_738_368.0,  // 2^35
    sigma_r: 0.5,
    ek_bytes: 13888, ct_bytes: 57056,
};
const PS_256_R: ParamSet = ParamSet {
    name: "k=256 robust",
    kappa: 256, robust: true,
    d: 4096, log_d: 12,
    sigma_s: 18.0,
    sigma_p: 268_435_456.0,     // 2^28
    sigma_r: 0.5,
    ek_bytes: 16448, ct_bytes: 61696,
};

const ALL_PARAMS: &[ParamSet] = &[PS_128_NR, PS_128_R, PS_256_NR, PS_256_R];
const PARTY_COUNTS: &[usize]  = &[4, 8, 16, 32];
const TARGET_LOG_Q: u32       = 50;

// =========================================================================
// Montgomery arithmetic over a ~50-bit prime modulus q (R = 2^64).
// =========================================================================

/// -q^{-1} mod 2^64, via Newton/Hensel lifting (doubles correct bits each step).
fn mont_q_neg_inv(q: u64) -> u64 {
    debug_assert!(q & 1 == 1, "q must be odd");
    let mut x: u64 = 1;
    for _ in 0..6 {
        x = x.wrapping_mul(2u64.wrapping_sub(q.wrapping_mul(x)));
    }
    debug_assert_eq!(q.wrapping_mul(x), 1);
    x.wrapping_neg()
}

/// R^2 mod q, where R = 2^64. Used to convert a standard u64 into Montgomery
/// form via one Montgomery multiplication.
fn mont_r2(q: u64) -> u64 {
    let r = ((1u128 << 64) % q as u128) as u64;
    ((r as u128 * r as u128) % q as u128) as u64
}

/// Montgomery reduction: given T < q * 2^64, return T * 2^{-64} mod q in [0, q).
#[inline(always)]
fn mont_redc(t: u128, q: u64, q_neg_inv: u64) -> u64 {
    let m: u64 = (t as u64).wrapping_mul(q_neg_inv);
    let t2: u128 = t.wrapping_add((m as u128).wrapping_mul(q as u128));
    let r = (t2 >> 64) as u64;
    if r >= q { r - q } else { r }
}

/// Montgomery multiplication. With both operands in Montgomery form, the
/// result is also in Montgomery form — the whole NTT can stay there.
#[inline(always)]
fn mont_mul(a: u64, b: u64, q: u64, q_neg_inv: u64) -> u64 {
    mont_redc((a as u128) * (b as u128), q, q_neg_inv)
}

// Plain "mod q" mul, only used at startup (Proth search, twiddle generation).
#[inline]
fn mulmod(a: u64, b: u64, q: u64) -> u64 {
    ((a as u128 * b as u128) % q as u128) as u64
}

fn powmod(mut base: u64, mut exp: u64, q: u64) -> u64 {
    let mut r: u64 = 1;
    base %= q;
    while exp > 0 {
        if exp & 1 == 1 { r = mulmod(r, base, q); }
        base = mulmod(base, base, q);
        exp >>= 1;
    }
    r
}

/// Deterministic Miller-Rabin for n < 2^64 using the 12-witness set.
fn is_prime(n: u64) -> bool {
    if n < 2 { return false; }
    const W: [u64; 12] = [2,3,5,7,11,13,17,19,23,29,31,37];
    for &p in &W {
        if n == p { return true; }
        if n % p == 0 { return false; }
    }
    let mut d = n - 1;
    let mut r: u32 = 0;
    while d & 1 == 0 { d >>= 1; r += 1; }
    'outer: for &a in &W {
        let mut x = powmod(a, d, n);
        if x == 1 || x == n - 1 { continue; }
        for _ in 0..r.saturating_sub(1) {
            x = mulmod(x, x, n);
            if x == n - 1 { continue 'outer; }
        }
        return false;
    }
    true
}

// =========================================================================
// NTT context, parameterised by ring degree d.
// =========================================================================

struct Ctx {
    d: usize,
    #[allow(dead_code)]
    log_d: u32,
    q: u64,
    q_neg_inv: u64,
    r2_mod_q: u64,
    psi_pow_mont:     Vec<u64>, // psi^{bitrev(k)} * R mod q
    psi_inv_pow_mont: Vec<u64>, // psi^{-bitrev(k)} * R mod q
    n_inv_mont: u64,            // d^{-1} * R mod q
}

fn bit_reverse(mut x: usize, bits: u32) -> usize {
    let mut r = 0usize;
    for _ in 0..bits { r = (r << 1) | (x & 1); x >>= 1; }
    r
}

impl Ctx {
    fn build(d: usize, log_d: u32) -> Self {
        assert_eq!(1usize << log_d, d);
        let two_d = 2 * d as u64;

        // 1) Proth prime q ≡ 1 (mod 2d), q ≥ 2^49.
        // k starts around 2^(TARGET_LOG_Q-1) / (2d), which for d=2048 is ~2^37,
        // so we bound the number of attempts (not the absolute value of k).
        let lo = 1u64 << (TARGET_LOG_Q - 1);
        let mut k = (lo + two_d - 1) / two_d;
        let mut tried: u64 = 0;
        let q = loop {
            let cand = k * two_d + 1;
            if is_prime(cand) { break cand; }
            k += 1;
            tried += 1;
            if tried > 1_000_000 {
                panic!("Proth-prime search exhausted near 2^{} after {} attempts",
                       TARGET_LOG_Q, tried);
            }
        };
        let q_neg_inv = mont_q_neg_inv(q);
        let r2_mod_q  = mont_r2(q);

        // 2) Primitive 2d-th root of unity psi with psi^d = -1.
        let exp = (q - 1) / two_d;
        let mut psi = 0u64;
        for g in 2u64.. {
            let cand = powmod(g, exp, q);
            if powmod(cand, d as u64, q) == q - 1 { psi = cand; break; }
            if g > 1 << 16 { panic!("psi search exhausted"); }
        }

        // 3) Twiddle tables in Montgomery + bit-reversed order.
        let psi_inv = powmod(psi, q - 2, q);
        let mut psi_lin     = vec![1u64; d];
        let mut psi_inv_lin = vec![1u64; d];
        for i in 1..d {
            psi_lin[i]     = mulmod(psi_lin[i - 1],     psi,     q);
            psi_inv_lin[i] = mulmod(psi_inv_lin[i - 1], psi_inv, q);
        }
        // x -> x * R mod q  (one Montgomery mul with R^2)
        let to_mont = |x: u64| mont_mul(x, r2_mod_q, q, q_neg_inv);
        let mut psi_pow_mont     = vec![0u64; d];
        let mut psi_inv_pow_mont = vec![0u64; d];
        for k in 0..d {
            let br = bit_reverse(k, log_d);
            psi_pow_mont[k]     = to_mont(psi_lin[br]);
            psi_inv_pow_mont[k] = to_mont(psi_inv_lin[br]);
        }
        let n_inv      = powmod(d as u64, q - 2, q);
        let n_inv_mont = to_mont(n_inv);

        Ctx { d, log_d, q, q_neg_inv, r2_mod_q, psi_pow_mont, psi_inv_pow_mont, n_inv_mont }
    }

    #[inline(always)] fn add(&self, a: u64, b: u64) -> u64 {
        let s = a + b; if s >= self.q { s - self.q } else { s }
    }
    #[inline(always)] fn sub(&self, a: u64, b: u64) -> u64 {
        if a >= b { a - b } else { a + self.q - b }
    }
    #[inline(always)] fn mm(&self, a: u64, b: u64) -> u64 {
        mont_mul(a, b, self.q, self.q_neg_inv)
    }

    fn to_mont(&self, x: u64) -> u64 { self.mm(x, self.r2_mod_q) }
    fn from_mont(&self, x: u64) -> u64 { mont_redc(x as u128, self.q, self.q_neg_inv) }

    /// Merged-psi forward NTT, Cooley-Tukey decimation-in-time, in place.
    /// Input/output in Montgomery form. Output is in bit-reversed order.
    fn ntt(&self, a: &mut [u64]) {
        let n = self.d;
        let q = self.q;
        let q_inv = self.q_neg_inv;
        let mut t = n;
        let mut m = 1usize;
        while m < n {
            t >>= 1;
            for i in 0..m {
                let s = self.psi_pow_mont[m + i];
                let j1 = 2 * i * t;
                // Hot inner loop. We hand-fuse the add/sub-with-reduce so
                // the compiler can keep everything in registers.
                for j in j1..(j1 + t) {
                    let u = a[j];
                    let v = mont_mul(a[j + t], s, q, q_inv);
                    let sum  = u + v;
                    a[j]     = if sum >= q { sum - q } else { sum };
                    a[j + t] = if u >= v { u - v } else { u + q - v };
                }
            }
            m <<= 1;
        }
    }

    /// Inverse NTT (Gentleman-Sande), with final n^{-1} scaling.
    fn intt(&self, a: &mut [u64]) {
        let n = self.d;
        let q = self.q;
        let q_inv = self.q_neg_inv;
        let mut t = 1usize;
        let mut m = n;
        while m > 1 {
            let h = m >> 1;
            let mut j1 = 0usize;
            for i in 0..h {
                let s = self.psi_inv_pow_mont[h + i];
                for j in j1..(j1 + t) {
                    let u = a[j];
                    let v = a[j + t];
                    let sum  = u + v;
                    a[j]     = if sum >= q { sum - q } else { sum };
                    let diff = if u >= v { u - v } else { u + q - v };
                    a[j + t] = mont_mul(diff, s, q, q_inv);
                }
                j1 += 2 * t;
            }
            t <<= 1;
            m = h;
        }
        let n_inv_m = self.n_inv_mont;
        for x in a.iter_mut() { *x = mont_mul(*x, n_inv_m, q, q_inv); }
    }

    /// Negacyclic poly mul. Inputs and output in Montgomery form.
    fn poly_mul(&self, a: &[u64], b: &[u64]) -> Vec<u64> {
        let mut an = a.to_vec();
        let mut bn = b.to_vec();
        self.ntt(&mut an);
        self.ntt(&mut bn);
        let q = self.q;
        let q_inv = self.q_neg_inv;
        for i in 0..self.d {
            an[i] = mont_mul(an[i], bn[i], q, q_inv);
        }
        self.intt(&mut an);
        an
    }

    /// Schoolbook reference (standard form, both inputs and output).
    /// Only used by the startup self-test.
    fn poly_mul_schoolbook_std(&self, a: &[u64], b: &[u64]) -> Vec<u64> {
        let d = self.d;
        let qz = self.q as i128;
        let mut c = vec![0i128; 2 * d];
        for i in 0..d {
            let ai = a[i] as i128;
            if ai == 0 { continue; }
            for j in 0..d { c[i + j] += ai * b[j] as i128; }
        }
        let mut out = vec![0u64; d];
        for i in 0..d {
            let v = (c[i] - c[i + d]) % qz;
            out[i] = if v < 0 { (v + qz) as u64 } else { v as u64 };
        }
        out
    }

    fn poly_add(&self, a: &[u64], b: &[u64]) -> Vec<u64> {
        (0..self.d).map(|i| self.add(a[i], b[i])).collect()
    }

    fn poly_zero(&self) -> Vec<u64> { vec![0u64; self.d] }

    /// Uniform polynomial directly in Montgomery form.
    fn poly_rand_uniform_mont(&self, rng: &mut StdRng) -> Vec<u64> {
        let q = self.q;
        (0..self.d).map(|_| self.to_mont(rng.gen_range(0..q))).collect()
    }

    /// Gaussian polynomial via Box-Muller PAIRS (two samples per draw).
    fn sample_poly_gauss_mont(&self, sigma: f64, rng: &mut StdRng) -> Vec<u64> {
        let qi = self.q as i64;
        let mut out = vec![0u64; self.d];
        let mut i = 0;
        while i < self.d {
            let u1: f64 = rng.gen_range(1e-12_f64..1.0);
            let u2: f64 = rng.gen_range(0.0_f64..1.0);
            let r = (-2.0 * u1.ln()).sqrt() * sigma;
            let theta = 2.0 * std::f64::consts::PI * u2;
            let g1 = (r * theta.cos()).round() as i64;
            let g2 = (r * theta.sin()).round() as i64;
            out[i] = self.to_mont(g1.rem_euclid(qi) as u64);
            if i + 1 < self.d {
                out[i + 1] = self.to_mont(g2.rem_euclid(qi) as u64);
            }
            i += 2;
        }
        out
    }
}

// =========================================================================
// Sasha2 phases. Operation counts follow the cited algorithms.
// =========================================================================

struct Setup {
    a: Vec<u64>, b: Vec<u64>, t: Vec<u64>,
}

fn tibe_setup(c: &Ctx, ps: &ParamSet, n_parties: usize, rng: &mut StdRng) -> Setup {
    let a = c.poly_rand_uniform_mont(rng);
    let s  = c.sample_poly_gauss_mont(ps.sigma_s, rng);
    let sp = c.sample_poly_gauss_mont(ps.sigma_s, rng);
    let as_ = c.poly_mul(&a, &s);
    let b   = c.poly_add(&as_, &sp);
    let t   = c.poly_rand_uniform_mont(rng);

    // (T, N) sharing of (s, s'): 2 * d coefficients evaluated at N points.
    // Each evaluation is one Horner step of length T = N-ish; conservatively
    // count 2 * d * N scalar mults. Tiny compared to the poly muls above.
    let mut sink: u64 = 0;
    for _ in 0..(2 * c.d * n_parties) {
        sink = sink.wrapping_add(c.mm(rng.gen_range(0..c.q), rng.gen_range(0..c.q)));
    }
    std::hint::black_box(sink);

    Setup { a, b, t }
}

fn tibe_encrypt(c: &Ctx, ps: &ParamSet, setup: &Setup, rng: &mut StdRng) {
    let h_id = c.poly_rand_uniform_mont(rng);          // Hid(id) modelled as URS
    let r    = c.sample_poly_gauss_mont(ps.sigma_r, rng);
    let _e0  = c.sample_poly_gauss_mont(ps.sigma_p, rng);
    let _e1  = c.sample_poly_gauss_mont(ps.sigma_p, rng);
    let _e2  = c.sample_poly_gauss_mont(ps.sigma_p, rng);
    let _ep  = c.sample_poly_gauss_mont(ps.sigma_p, rng);
    let u0 = c.poly_mul(&setup.a, &r);
    let u1 = c.poly_mul(&setup.b, &r);
    let u2 = c.poly_mul(&h_id,    &r);
    let v  = c.poly_mul(&setup.t, &r);
    std::hint::black_box((u0, u1, u2, v));
}

fn share_extract_one(c: &Ctx, ps: &ParamSet, setup: &Setup, rng: &mut StdRng) {
    let h_id = c.poly_rand_uniform_mont(rng);
    // Round 0: sample p^(i) in R^4 (the third coord is zero per Alg. 10),
    // compute w^(i) = [1, a, b, H_id] · p^(i) -> 3 poly muls.
    let p0  = c.sample_poly_gauss_mont(ps.sigma_p, rng);
    let p1  = c.sample_poly_gauss_mont(ps.sigma_p, rng);
    let _p2 = c.poly_zero();
    let p3  = c.sample_poly_gauss_mont(SIGMA_P_PRIME, rng);
    let w_a   = c.poly_mul(&setup.a, &p1);
    let w_b   = c.poly_mul(&setup.b, &p1);
    let w_hid = c.poly_mul(&h_id,    &p3);
    let _w = c.poly_add(&c.poly_add(&p0, &w_a), &c.poly_add(&w_b, &w_hid));
    // Round 2: z^(i) = p^(i) + c0 · s^(i) -> 2 poly muls (s^(i) has 2 nonzero coords).
    let c0   = c.poly_rand_uniform_mont(rng);
    let s_i  = c.sample_poly_gauss_mont(ps.sigma_s, rng);
    let sp_i = c.sample_poly_gauss_mont(ps.sigma_s, rng);
    let cs0 = c.poly_mul(&c0, &s_i);
    let cs1 = c.poly_mul(&c0, &sp_i);
    let z0 = c.poly_add(&p0, &cs0);
    let z1 = c.poly_add(&p1, &cs1);
    std::hint::black_box((z0, z1));
}

fn combine_one(c: &Ctx, ps: &ParamSet, n_parties: usize, setup: &Setup, rng: &mut StdRng) {
    let h_id = c.poly_rand_uniform_mont(rng);
    // Robust variant only: N calls to ShareVerify, each 3 poly muls for the
    // [1, a, b, H_id] · z^(i) == w^(i) + c0 · b^(i) equality assert.
    if ps.robust {
        for _ in 0..n_parties {
            let z0 = c.poly_rand_uniform_mont(rng);
            let z1 = c.poly_rand_uniform_mont(rng);
            let z3 = c.poly_rand_uniform_mont(rng);
            let _va = c.poly_mul(&setup.a, &z0);
            let _vb = c.poly_mul(&setup.b, &z1);
            let _vh = c.poly_mul(&h_id,    &z3);
        }
    }
    // Final reconstruction: v - u^T · z' with z' having 3 coords -> 3 poly muls.
    let z0p = c.poly_rand_uniform_mont(rng);
    let z1p = c.poly_rand_uniform_mont(rng);
    let z3p = c.poly_rand_uniform_mont(rng);
    let _f0 = c.poly_mul(&setup.a, &z0p);
    let _f1 = c.poly_mul(&setup.b, &z1p);
    let _f2 = c.poly_mul(&h_id,    &z3p);
}

// =========================================================================
// Driver
// =========================================================================

#[derive(Clone, Copy)]
struct Timing {
    setup: Duration,
    encaps: Duration,
    share: Duration,
    combine: Duration,
}

fn run_one(c: &Ctx, ps: &ParamSet, n_parties: usize, rng: &mut StdRng) -> Timing {
    let t0 = Instant::now();
    let setup_ = tibe_setup(c, ps, n_parties, rng);
    let setup = t0.elapsed();

    let t0 = Instant::now();
    tibe_encrypt(c, ps, &setup_, rng);
    let encaps = t0.elapsed();

    let t0 = Instant::now();
    share_extract_one(c, ps, &setup_, rng);
    let share = t0.elapsed();

    let t0 = Instant::now();
    combine_one(c, ps, n_parties, &setup_, rng);
    let combine = t0.elapsed();

    Timing { setup, encaps, share, combine }
}

fn fmt_d(d: Duration) -> String {
    let us = d.as_secs_f64() * 1_000_000.0;
    if us >= 1_000_000.0 { format!("{:>7.2} s",  us / 1_000_000.0) }
    else if us >= 1000.0 { format!("{:>7.2} ms", us / 1000.0) }
    else { format!("{:>7.2} us", us) }
}

fn main() {
    println!("===== sasha2 cost estimate (Boudgoust et al., PKC'26) =====");
    println!();

    // Build NTT contexts (one per ring degree, amortised across all 16 runs).
    let t0 = Instant::now();
    let ctx_2048 = Ctx::build(2048, 11);
    let ctx_4096 = Ctx::build(4096, 12);
    let build_t = t0.elapsed();
    println!("NTT contexts built in {} (one-time, amortised over all configs):", fmt_d(build_t));
    println!("  d = 2048 (kappa=128) : q = {} (~2^{:.3})", ctx_2048.q, (ctx_2048.q as f64).log2());
    println!("  d = 4096 (kappa=256) : q = {} (~2^{:.3})", ctx_4096.q, (ctx_4096.q as f64).log2());
    println!();

    // Self-test: NTT mul == schoolbook mul on random std-form polys.
    let mut rng = StdRng::from_entropy();
    for c in [&ctx_2048, &ctx_4096] {
        let a_std: Vec<u64> = (0..c.d).map(|_| rng.gen_range(0..c.q)).collect();
        let b_std: Vec<u64> = (0..c.d).map(|_| rng.gen_range(0..c.q)).collect();
        let a_m: Vec<u64> = a_std.iter().map(|&x| c.to_mont(x)).collect();
        let b_m: Vec<u64> = b_std.iter().map(|&x| c.to_mont(x)).collect();
        let ntt_out_m = c.poly_mul(&a_m, &b_m);
        let ntt_out_std: Vec<u64> = ntt_out_m.iter().map(|&x| c.from_mont(x)).collect();
        let sb_out = c.poly_mul_schoolbook_std(&a_std, &b_std);
        assert_eq!(ntt_out_std, sb_out, "NTT vs schoolbook mismatch at d = {}", c.d);
    }
    println!("Self-test (NTT == schoolbook) : OK for both d = 2048 and d = 4096");
    println!();

    // Per-mul calibration.
    println!("Per-mul calibration (Montgomery NTT):");
    for c in [&ctx_2048, &ctx_4096] {
        let a = c.poly_rand_uniform_mont(&mut rng);
        let b = c.poly_rand_uniform_mont(&mut rng);
        const N_CAL: u32 = 100;
        let t0 = Instant::now();
        for _ in 0..N_CAL { std::hint::black_box(c.poly_mul(&a, &b)); }
        let pmul = t0.elapsed() / N_CAL;
        println!("  d = {:>4} : {} / poly mul", c.d, fmt_d(pmul));
    }
    println!();

    // Sweep: all 4 parameter sets x {4, 8, 16, 32} parties.
    println!("=====================================================================================");
    println!("                Per-phase timings  (Montgomery NTT, single-thread)");
    println!("=====================================================================================");
    println!();
    println!("{:<18} {:>3} {:>10} {:>10} {:>10} {:>10} {:>11} {:>11}",
             "config", "N", "Setup", "Encaps", "Share/p", "Combine", "Decap(seq)", "Decap(par)");
    println!("{}", "-".repeat(89));

    let mut summary: Vec<(ParamSet, usize, Timing)> = Vec::new();
    for ps in ALL_PARAMS {
        let c = if ps.d == 2048 { &ctx_2048 } else { &ctx_4096 };
        for &n in PARTY_COUNTS {
            let t = run_one(c, ps, n, &mut rng);
            let total_seq = t.share * n as u32 + t.combine;
            let total_par = t.share + t.combine;
            println!("{:<18} {:>3} {:>10} {:>10} {:>10} {:>10} {:>11} {:>11}",
                     ps.name, n,
                     fmt_d(t.setup), fmt_d(t.encaps),
                     fmt_d(t.share), fmt_d(t.combine),
                     fmt_d(total_seq), fmt_d(total_par));
            summary.push((*ps, n, t));
        }
        println!();
    }

    // Sizes table for context.
    println!("Sasha2 sizes (sasha2 Table 1, paper Eqs. 40-41):");
    println!("  {:<18}  |ek|  {:>8} B   |ct|  {:>8} B", PS_128_NR.name, PS_128_NR.ek_bytes, PS_128_NR.ct_bytes);
    println!("  {:<18}  |ek|  {:>8} B   |ct|  {:>8} B", PS_128_R.name,  PS_128_R.ek_bytes,  PS_128_R.ct_bytes);
    println!("  {:<18}  |ek|  {:>8} B   |ct|  {:>8} B", PS_256_NR.name, PS_256_NR.ek_bytes, PS_256_NR.ct_bytes);
    println!("  {:<18}  |ek|  {:>8} B   |ct|  {:>8} B", PS_256_R.name,  PS_256_R.ek_bytes,  PS_256_R.ct_bytes);
    println!();

    println!("Legend:");
    println!("  Share/p     = one party's ShareExtract (rounds 0+1+2)");
    println!("  Decap(seq)  = N * Share/p + Combine  (parties act sequentially)");
    println!("  Decap(par)  = max Share/p + Combine  (parties act in parallel, then Combine)");
    println!();
    println!("Notes:");
    println!("  * NTT poly mul (Montgomery, R = 2^64); single-thread");
    println!("  * Robust Combine includes N ShareVerify calls (3 poly muls each)");
    println!("  * Non-robust Combine does only the 3-mul final reconstruction");
    println!("  * Excludes ML-DSA / Falcon signature in BCHK+ (~50-500 us)");
    println!("  * Excludes SHA3/SHAKE random-oracle calls (~tens of us each)");
    println!("  * Vandermonde sharing approximated as 2 * d * N scalar mults");
    println!("  * AVX2 / SIMD butterflies would give another ~2-3x on x86-64");
}