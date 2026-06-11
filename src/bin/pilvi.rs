//! Cost estimator for **Pilvi** = Cini, Lai, Woo,
//! "Pilvi: Lattice Threshold PKE with Small Decryption Shares and Improved
//! Security", ePrint 2025/1691.
//!
//! Sweeps the 8 parameter sets reported in Table 5:
//!   Pilvi1792-2-8-1    (K= 8, t= 2, Q=1,    q ~ 2^56)
//!   Pilvi2048-6-8-1    (K= 8, t= 6, Q=1,    q ~ 2^63)
//!   Pilvi2304-10-16-1  (K=16, t=10, Q=1,    q ~ 2^70)
//!   Pilvi2816-16-32-1  (K=32, t=16, Q=1,    q ~ 2^84)
//!   Pilvi3072-2-8-x60  (K= 8, t= 2, Q=2^60, q ~ 2^89)
//!   Pilvi3072-6-8-x60  (K= 8, t= 6, Q=2^60, q ~ 2^94)
//!   Pilvi3584-10-16-x60(K=16, t=10, Q=2^60, q ~ 2^102)
//!   Pilvi3840-16-32-x60(K=32, t=16, Q=2^60, q ~ 2^115)
//!
//! Implementation strategy: Micciancio-Suhl style + ring kernel
//! ------------------------------------------------------------
//! Pilvi is a thresholdised Regev PKE over the cyclotomic ring
//! R_q = Z_q[X]/(X^256 + 1)  (f = 512, phi = 256). Every protocol step is a
//! linear-algebra operation over R_q — matrix-vector products, inner products,
//! linear combinations — exactly the Micciancio-Suhl pattern, with one twist:
//! each "scalar" is now a ring element in R_q.
//!
//! So the cost model is:
//!
//!   matvec(R_q^{n x m}, R_q^m)      = n*m  ring mults + n*(m-1)  ring adds
//!   inner_product(R_q^n, R_q^n)     = n    ring mults + (n-1)    ring adds
//!   Vandermonde share (K x t) (R^{t x n})  = K*t*n   ring mults
//!
//! Ring multiplication is via negacyclic NTT at degree d = 256. The modulus q
//! ranges from ~2^25 (Q=0 case, not shown here) to ~2^115 (Q=2^60, K=32),
//! which doesn't fit in u64. We use the same RNS+Montgomery pattern as the
//! sasha1 estimator: pick k Proth primes of ~50 bits each, do k independent
//! single-prime NTTs per ring mul, then a CRT reconstruction sink.
//!
//! The number of RNS primes is set per parameter set so that the product
//! prod_i q_i comfortably exceeds q with headroom for intermediate
//! convolution sums:
//!
//!   log q  <=  50  -> k = 1   (Q = 0 regime; not in Table 5 but supported)
//!   log q  <=  85  -> k = 2
//!   log q  <= 130  -> k = 3
//!
//! Build instructions
//! ------------------
//!   Drop into examples/bench_pilvi.rs (or src/bin/), then:
//!     cargo run --release --example bench_pilvi
//!
//! Notes on the cost model
//! -----------------------
//!   * ParDec is single-round, non-interactive: each party emits one length-n
//!     ring inner product + one Gaussian ring element. Output is a single
//!     ring element (~phi*log q / 8 bytes; 1.7-3.6 KB depending on params).
//!     This makes Pilvi very close to Micciancio-Suhl operationally, modulo
//!     the upgrade from u32 scalar to R_q ring scalar.
//!   * Combine is a length-t linear combination with coefficients from
//!     v_0^T V_T^{-1} (norm rho(t) ~ 91-264 for the parameter sets in Table 5).
//!     These coefficients are *ring* elements (not scalar ints), so each
//!     summand is a full ring mul. t is small (2 to 16 in Table 5) so combine
//!     is cheap.
//!   * KGen is dominated by the (K x t)-by-(t x n) Vandermonde share product:
//!     K*t*n ring mults. Entries of V are low-norm roots of unity. We count
//!     them as full ring mults (conservative upper bound); a multiply-by-zeta
//!     optimisation would cut this further.
//!   * L = ceil(2*lambda/phi) = ceil(256/256) = 1 for all parameter sets in
//!     Table 5, so there is exactly one (r, b, c_1, pd) per ciphertext —
//!     not L independent copies.
//!   * Excludes: SHAKE/SHA3 for sampling A from a seed (negligible),
//!     bounded-uniform vs Gaussian noise (same cost), RLWE-secret-key
//!     serialisation (~10 us range).

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::time::{Duration, Instant};

// =========================================================================
// Pilvi parameter sets (Table 5).
// =========================================================================

#[derive(Clone, Copy)]
struct ParamSet {
    name: &'static str,
    k_parties: usize,    // K, the maximum number of users
    t_thresh: usize,     // recovery threshold t
    n: usize,            // module rank (height of A)
    log_q: u32,          // bits in modulus q
    q_log_label: &'static str,
    rho_t: u64,          // recovery-expansion factor (cosmetic; used for combine ring-coeff norm)
    ct_kb: f64,          // |ctxt| in KB (from Table 5)
    pd_kb: f64,          // |pdk|  in KB
    q_descr: &'static str,
}

// Eight rows from Table 5. ell = ceil(256/phi) = 1 for all.
// rho(t) values pulled from Appendix D.1 console output:
//   K=8:   rho(2)=46, rho(6)=91
//   K=16:  rho(10)=678
//   K=32:  rho(16)=63908
const PARAMS: &[ParamSet] = &[
    ParamSet { name: "Pilvi1792-2-8-1",     k_parties: 8,  t_thresh:  2, n:  7, log_q: 56,  q_log_label: "2^56",  rho_t: 46,    ct_kb: 14.0, pd_kb: 1.7, q_descr: "Q=1" },
    ParamSet { name: "Pilvi2048-6-8-1",     k_parties: 8,  t_thresh:  6, n:  8, log_q: 63,  q_log_label: "2^63",  rho_t: 91,    ct_kb: 17.5, pd_kb: 1.9, q_descr: "Q=1" },
    ParamSet { name: "Pilvi2304-10-16-1",   k_parties: 16, t_thresh: 10, n:  9, log_q: 70,  q_log_label: "2^70",  rho_t: 678,   ct_kb: 21.8, pd_kb: 2.2, q_descr: "Q=1" },
    ParamSet { name: "Pilvi2816-16-32-1",   k_parties: 32, t_thresh: 16, n: 11, log_q: 84,  q_log_label: "2^84",  rho_t: 63908, ct_kb: 31.1, pd_kb: 2.6, q_descr: "Q=1" },
    ParamSet { name: "Pilvi3072-2-8-x60",   k_parties: 8,  t_thresh:  2, n: 12, log_q: 89,  q_log_label: "2^89",  rho_t: 46,    ct_kb: 35.8, pd_kb: 2.8, q_descr: "Q=2^60" },
    ParamSet { name: "Pilvi3072-6-8-x60",   k_parties: 8,  t_thresh:  6, n: 12, log_q: 94,  q_log_label: "2^94",  rho_t: 91,    ct_kb: 38.1, pd_kb: 2.9, q_descr: "Q=2^60" },
    ParamSet { name: "Pilvi3584-10-16-x60", k_parties: 16, t_thresh: 10, n: 14, log_q: 102, q_log_label: "2^102", rho_t: 678,   ct_kb: 47.6, pd_kb: 3.2, q_descr: "Q=2^60" },
    ParamSet { name: "Pilvi3840-16-32-x60", k_parties: 32, t_thresh: 16, n: 15, log_q: 115, q_log_label: "2^115", rho_t: 63908, ct_kb: 57.1, pd_kb: 3.6, q_descr: "Q=2^60" },
];

// Ring degree d = phi = 256, m = 2n + ell with ell = 1.
const D: usize          = 256;
const LOG_D: u32        = 8;          // log2(256)
const ELL: usize        = 1;          // message blocks (256 bits / phi=256)
const TARGET_LOG_Q: u32 = 50;         // per RNS prime
const SIGMA_X: f64      = 6.0;        // encryption-randomness Gaussian; coarse
const SIGMA_E: f64      = 6.0;        // public-key error
const SIGMA_SM: f64     = 6.0;        // partial-decryption smudging
// (the exact sigmas affect only sampling time, not the linear-algebra cost)

/// Choose RNS prime count from log q. See header comment for the breakdown.
fn rns_primes_for(log_q: u32) -> usize {
    if log_q <= 50 { 1 } else if log_q <= 85 { 2 } else { 3 }
}

// =========================================================================
// Montgomery arithmetic over a ~50-bit prime (one RNS component).
// Lifted verbatim from the sasha2 estimator.
// =========================================================================

fn mont_q_neg_inv(q: u64) -> u64 {
    debug_assert!(q & 1 == 1);
    let mut x: u64 = 1;
    for _ in 0..6 { x = x.wrapping_mul(2u64.wrapping_sub(q.wrapping_mul(x))); }
    debug_assert_eq!(q.wrapping_mul(x), 1);
    x.wrapping_neg()
}

fn mont_r2(q: u64) -> u64 {
    let r = ((1u128 << 64) % q as u128) as u64;
    ((r as u128 * r as u128) % q as u128) as u64
}

#[inline(always)]
fn mont_redc(t: u128, q: u64, q_neg_inv: u64) -> u64 {
    let m: u64 = (t as u64).wrapping_mul(q_neg_inv);
    let t2: u128 = t.wrapping_add((m as u128).wrapping_mul(q as u128));
    let r = (t2 >> 64) as u64;
    if r >= q { r - q } else { r }
}

#[inline(always)]
fn mont_mul(a: u64, b: u64, q: u64, q_neg_inv: u64) -> u64 {
    mont_redc((a as u128) * (b as u128), q, q_neg_inv)
}

#[inline] fn mulmod(a: u64, b: u64, q: u64) -> u64 {
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

fn is_prime(n: u64) -> bool {
    if n < 2 { return false; }
    const W: [u64; 12] = [2,3,5,7,11,13,17,19,23,29,31,37];
    for &p in &W { if n == p { return true; } if n % p == 0 { return false; } }
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
// NTT context for one RNS component at degree d = 256.
// =========================================================================

struct Ctx {
    d: usize,
    q: u64,
    q_neg_inv: u64,
    r2_mod_q: u64,
    psi_pow_mont: Vec<u64>,
    psi_inv_pow_mont: Vec<u64>,
    n_inv_mont: u64,
}

fn bit_reverse(mut x: usize, bits: u32) -> usize {
    let mut r = 0usize;
    for _ in 0..bits { r = (r << 1) | (x & 1); x >>= 1; }
    r
}

impl Ctx {
    fn build(d: usize, log_d: u32, prime_skip: u64) -> Self {
        let two_d = 2 * d as u64;
        let lo = 1u64 << (TARGET_LOG_Q - 1);
        let mut k = (lo + two_d - 1) / two_d + prime_skip;  // skip lets us pick distinct primes per RNS slot
        let mut tried: u64 = 0;
        let q = loop {
            let cand = k * two_d + 1;
            if is_prime(cand) { break cand; }
            k += 1;
            tried += 1;
            if tried > 1_000_000 { panic!("Proth-prime search exhausted"); }
        };
        let q_neg_inv = mont_q_neg_inv(q);
        let r2_mod_q  = mont_r2(q);

        let exp = (q - 1) / two_d;
        let mut psi = 0u64;
        for g in 2u64.. {
            let cand = powmod(g, exp, q);
            if powmod(cand, d as u64, q) == q - 1 { psi = cand; break; }
            if g > 1 << 16 { panic!("psi search exhausted"); }
        }
        let psi_inv = powmod(psi, q - 2, q);
        let mut psi_lin     = vec![1u64; d];
        let mut psi_inv_lin = vec![1u64; d];
        for i in 1..d {
            psi_lin[i]     = mulmod(psi_lin[i - 1],     psi,     q);
            psi_inv_lin[i] = mulmod(psi_inv_lin[i - 1], psi_inv, q);
        }
        let to_mont = |x: u64| mont_mul(x, r2_mod_q, q, q_neg_inv);
        let mut psi_pow_mont     = vec![0u64; d];
        let mut psi_inv_pow_mont = vec![0u64; d];
        for k in 0..d {
            let br = bit_reverse(k, log_d);
            psi_pow_mont[k]     = to_mont(psi_lin[br]);
            psi_inv_pow_mont[k] = to_mont(psi_inv_lin[br]);
        }
        let n_inv = powmod(d as u64, q - 2, q);
        let n_inv_mont = to_mont(n_inv);
        Ctx { d, q, q_neg_inv, r2_mod_q, psi_pow_mont, psi_inv_pow_mont, n_inv_mont }
    }

    #[inline(always)] fn add(&self, a: u64, b: u64) -> u64 {
        let s = a + b; if s >= self.q { s - self.q } else { s }
    }
    #[inline(always)] fn mm(&self, a: u64, b: u64) -> u64 {
        mont_mul(a, b, self.q, self.q_neg_inv)
    }
    fn to_mont(&self, x: u64) -> u64 { self.mm(x, self.r2_mod_q) }

    fn ntt(&self, a: &mut [u64]) {
        let n = self.d; let q = self.q; let q_inv = self.q_neg_inv;
        let mut t = n; let mut m = 1usize;
        while m < n {
            t >>= 1;
            for i in 0..m {
                let s = self.psi_pow_mont[m + i];
                let j1 = 2 * i * t;
                for j in j1..(j1 + t) {
                    let u = a[j];
                    let v = mont_mul(a[j + t], s, q, q_inv);
                    let sum = u + v;
                    a[j]     = if sum >= q { sum - q } else { sum };
                    a[j + t] = if u >= v { u - v } else { u + q - v };
                }
            }
            m <<= 1;
        }
    }
    fn intt(&self, a: &mut [u64]) {
        let n = self.d; let q = self.q; let q_inv = self.q_neg_inv;
        let mut t = 1usize; let mut m = n;
        while m > 1 {
            let h = m >> 1;
            let mut j1 = 0usize;
            for i in 0..h {
                let s = self.psi_inv_pow_mont[h + i];
                for j in j1..(j1 + t) {
                    let u = a[j]; let v = a[j + t];
                    let sum = u + v;
                    a[j]     = if sum >= q { sum - q } else { sum };
                    let diff = if u >= v { u - v } else { u + q - v };
                    a[j + t] = mont_mul(diff, s, q, q_inv);
                }
                j1 += 2 * t;
            }
            t <<= 1; m = h;
        }
        let n_inv_m = self.n_inv_mont;
        for x in a.iter_mut() { *x = mont_mul(*x, n_inv_m, q, q_inv); }
    }

    /// Single-prime negacyclic poly mul (one RNS component). Building block.
    fn poly_mul_single(&self, a: &[u64], b: &[u64]) -> Vec<u64> {
        let mut an = a.to_vec();
        let mut bn = b.to_vec();
        self.ntt(&mut an); self.ntt(&mut bn);
        let q = self.q; let q_inv = self.q_neg_inv;
        for i in 0..self.d { an[i] = mont_mul(an[i], bn[i], q, q_inv); }
        self.intt(&mut an);
        an
    }

    fn poly_add(&self, a: &[u64], b: &[u64]) -> Vec<u64> {
        (0..self.d).map(|i| self.add(a[i], b[i])).collect()
    }
    fn poly_zero(&self) -> Vec<u64> { vec![0u64; self.d] }
    fn poly_rand_uniform_mont(&self, rng: &mut StdRng) -> Vec<u64> {
        let q = self.q;
        (0..self.d).map(|_| self.to_mont(rng.gen_range(0..q))).collect()
    }
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
            if i + 1 < self.d { out[i + 1] = self.to_mont(g2.rem_euclid(qi) as u64); }
            i += 2;
        }
        out
    }
}

// =========================================================================
// RNS-NTT ring multiplication: the kernel for every linear-algebra op.
//
// Each "ring mul" is k single-prime NTT muls (k = 1, 2, or 3 depending on
// the parameter set's log q), plus a CRT reconstruction sink. The first
// result is the only one with semantically meaningful values; the others
// are timing-only (real RNS uses distinct Proth primes with distinct
// twiddle tables, but cost per NTT is independent of which prime is used).
// =========================================================================

fn ring_mul(ctxs: &[Ctx], a: &[u64], b: &[u64]) -> Vec<u64> {
    let r0 = ctxs[0].poly_mul_single(a, b);
    for c in &ctxs[1..] {
        let r_i = c.poly_mul_single(a, b);
        std::hint::black_box(&r_i);
    }
    if ctxs.len() > 1 {
        // CRT reconstruction: ~k * D u128 ops per coefficient.
        let c0 = &ctxs[0];
        let mut crt_sink: u64 = 0;
        for i in 0..D {
            for _ in 0..ctxs.len() {
                crt_sink = crt_sink.wrapping_add(
                    mont_mul(r0[i], c0.r2_mod_q, c0.q, c0.q_neg_inv));
            }
        }
        std::hint::black_box(crt_sink);
    }
    r0
}

// =========================================================================
// Linear-algebra primitives over R_q (the Micciancio-Suhl pattern).
//
// Each function takes a slice of RNS NTT contexts so the ring-mul cost
// scales correctly with log q.
// =========================================================================

/// Matrix-vector product: y = A * x where A is rows x cols (row-major,
/// each entry a ring element of D coefficients), x is a length-cols vector
/// of ring elements. Output is rows ring elements.
/// Cost: rows * cols ring muls + rows * (cols - 1) ring adds.
fn ring_matvec(ctxs: &[Ctx], a_flat: &[Vec<u64>], x: &[Vec<u64>],
               rows: usize, cols: usize) -> Vec<Vec<u64>> {
    let c0 = &ctxs[0];
    let mut y: Vec<Vec<u64>> = (0..rows).map(|_| c0.poly_zero()).collect();
    for i in 0..rows {
        for j in 0..cols {
            let prod = ring_mul(ctxs, &a_flat[i * cols + j], &x[j]);
            y[i] = c0.poly_add(&y[i], &prod);
        }
    }
    y
}

/// Row-vector times matrix: y^T = r^T * A. Same cost as matvec.
/// Used for b^T = r^T * A in KeyGen.
fn ring_vecmat(ctxs: &[Ctx], r: &[Vec<u64>], a_flat: &[Vec<u64>],
               rows: usize, cols: usize) -> Vec<Vec<u64>> {
    let c0 = &ctxs[0];
    let mut y: Vec<Vec<u64>> = (0..cols).map(|_| c0.poly_zero()).collect();
    for j in 0..cols {
        for i in 0..rows {
            let prod = ring_mul(ctxs, &r[i], &a_flat[i * cols + j]);
            y[j] = c0.poly_add(&y[j], &prod);
        }
    }
    y
}

/// Inner product of two length-n ring-element vectors.
/// Cost: n ring muls + (n - 1) ring adds. Output: one ring element.
fn ring_inner_product(ctxs: &[Ctx], a: &[Vec<u64>], b: &[Vec<u64>]) -> Vec<u64> {
    let c0 = &ctxs[0];
    let mut acc = c0.poly_zero();
    for i in 0..a.len() {
        let prod = ring_mul(ctxs, &a[i], &b[i]);
        acc = c0.poly_add(&acc, &prod);
    }
    acc
}

// =========================================================================
// Pilvi phases (Figure 5).
//
// L = 1 throughout (single 256-bit message slot), so there is one r, one b,
// one c_1, one pd per party. The K shares (s_k) are K ring-element vectors
// each of length n, packed in a row-major flat layout.
// =========================================================================

struct PublicKey {
    a:     Vec<Vec<u64>>,   // n*m ring elements (row-major)
    b:     Vec<Vec<u64>>,   // m ring elements
}

struct Shares {
    // K shares, each a length-n vector of ring elements. Flat: k*n + i.
    s: Vec<Vec<u64>>,
}

/// KGen: A is just sampled (already in `pp`), so we only count the
/// per-key work: b^T = r^T * A + e^T, plus the Vandermonde share product.
fn keygen(ctxs: &[Ctx], ps: &ParamSet, rng: &mut StdRng) -> (PublicKey, Shares) {
    let c0 = &ctxs[0];
    let n = ps.n;
    let m = 2 * n + ELL;
    let k = ps.k_parties;
    let t = ps.t_thresh;

    // A in R_q^{n x m}, uniform. (In a real deployment this is regenerated
    // from a 32-byte seed via SHAKE; the cost is ~ms range and not on the
    // critical path. Counted here as raw uniform sampling.)
    let a_flat: Vec<Vec<u64>> = (0..n * m).map(|_| c0.poly_rand_uniform_mont(rng)).collect();

    // R in R_q^{t x n}, with first row r^T (the LWE secret). Uniform.
    // We keep the whole R for the share product below.
    let r_mat: Vec<Vec<u64>> = (0..t * n).map(|_| c0.poly_rand_uniform_mont(rng)).collect();
    let r_row: Vec<Vec<u64>> = (0..n).map(|j| r_mat[j].clone()).collect();  // r^T = first row

    // e in R_q^m, Gaussian.
    let e: Vec<Vec<u64>> = (0..m).map(|_| c0.sample_poly_gauss_mont(SIGMA_E, rng)).collect();

    // b^T = r^T * A + e^T   (1 x n) * (n x m) = (1 x m)
    let mut b = ring_vecmat(ctxs, &r_row, &a_flat, n, m);
    for j in 0..m { b[j] = c0.poly_add(&b[j], &e[j]); }

    // Shares: S = V * R where V is K x t (Vandermonde of subtractive set),
    // R is t x n. Each entry of S is a sum of t (ring-scalar * ring-element)
    // products. V's entries are roots of unity (norm 1) so an optimised
    // implementation would do these as rotations, but we count full ring
    // mults for safety. Total: K * t * n ring mults.
    let mut s_flat: Vec<Vec<u64>> = (0..k * n).map(|_| c0.poly_zero()).collect();
    // V is fixed by the subtractive set, model entries as uniform-mont
    // ring elements (cost-equivalent to fetching from a precomputed table).
    let v_mat: Vec<Vec<u64>> = (0..k * t).map(|_| c0.poly_rand_uniform_mont(rng)).collect();
    for kk in 0..k {
        for col in 0..n {
            let mut acc = c0.poly_zero();
            for row in 0..t {
                let prod = ring_mul(ctxs, &v_mat[kk * t + row], &r_mat[row * n + col]);
                acc = c0.poly_add(&acc, &prod);
            }
            s_flat[kk * n + col] = acc;
        }
    }

    (PublicKey { a: a_flat, b }, Shares { s: s_flat })
}

/// Enc: c_0 = A * x;   c_1 = b^T * x + ξ^{-1} * msg * floor(q/2).
fn encrypt(ctxs: &[Ctx], ps: &ParamSet, pk: &PublicKey, rng: &mut StdRng)
           -> (Vec<Vec<u64>>, Vec<u64>)
{
    let c0 = &ctxs[0];
    let n = ps.n;
    let m = 2 * n + ELL;

    // x in R^m, short Gaussian.
    let x: Vec<Vec<u64>> = (0..m).map(|_| c0.sample_poly_gauss_mont(SIGMA_X, rng)).collect();

    // c_0 = A * x        (n x m) * (m x 1) = (n x 1)
    let c_0 = ring_matvec(ctxs, &pk.a, &x, n, m);

    // c_1 = b^T * x      length-m inner product of ring elements
    let c_1 = ring_inner_product(ctxs, &pk.b, &x);
    // (the + msg * ξ^{-1} * floor(q/2) is a single poly_add by a known
    // constant; negligible.)

    (c_0, c_1)
}

/// ParDec for one party: pd_k = s_k^T * c_0 + e_k.
/// This is the inner loop the Micciancio-Suhl scheme reduces to,
/// generalised to ring scalars. Output: one ring element.
fn par_dec(ctxs: &[Ctx], ps: &ParamSet, s_k: &[Vec<u64>], c_0: &[Vec<u64>],
           rng: &mut StdRng) -> Vec<u64>
{
    let c0 = &ctxs[0];
    let n = ps.n;
    debug_assert_eq!(s_k.len(), n);
    debug_assert_eq!(c_0.len(), n);

    let dot = ring_inner_product(ctxs, s_k, c_0);
    let e_k = c0.sample_poly_gauss_mont(SIGMA_SM, rng);
    c0.poly_add(&dot, &e_k)
}

/// Combine: pd = v_0^T * V_T^{-1} * (pd_k)_{k in T}. Length-t linear
/// combination with ring-element coefficients (norm bounded by rho(t)).
/// Then y = (c_1 - pd) * ξ, decode.
fn combine(ctxs: &[Ctx], ps: &ParamSet, partials: &[Vec<u64>], _c_1: &[u64],
           rng: &mut StdRng) -> Vec<u64>
{
    let c0 = &ctxs[0];
    let t = ps.t_thresh;
    debug_assert_eq!(partials.len(), t);

    // The combine coefficients v_0^T * V_T^{-1} are ring elements determined
    // by the subtractive set; sample uniform-mont as cost proxy.
    let coeffs: Vec<Vec<u64>> = (0..t).map(|_| c0.poly_rand_uniform_mont(rng)).collect();
    let pd = ring_inner_product(ctxs, &coeffs, partials);
    // The (c_1 - pd) * ξ and decode steps are scalar / per-coefficient
    // rounding: O(d) operations, negligible compared to one ring mul.
    pd
}

// =========================================================================
// Driver
// =========================================================================

#[derive(Clone, Copy)]
struct Timing {
    setup:   Duration,
    encaps:  Duration,
    share:   Duration,
    combine: Duration,
}

fn run_one(ctxs: &[Ctx], ps: &ParamSet, rng: &mut StdRng) -> Timing {
    let t0 = Instant::now();
    let (pk, shares) = keygen(ctxs, ps, rng);
    let setup = t0.elapsed();

    let t0 = Instant::now();
    let (c_0, c_1) = encrypt(ctxs, ps, &pk, rng);
    let encaps = t0.elapsed();

    // Time one party's partial decryption.
    let s_k: Vec<Vec<u64>> = (0..ps.n).map(|i| shares.s[i].clone()).collect();
    let t0 = Instant::now();
    std::hint::black_box(par_dec(ctxs, ps, &s_k, &c_0, rng));
    let share = t0.elapsed();

    // Build t partial decryptions for combine input.
    let partials: Vec<Vec<u64>> = (0..ps.t_thresh).map(|k| {
        let s_k_kk: Vec<Vec<u64>> = (0..ps.n).map(|i| shares.s[k * ps.n + i].clone()).collect();
        par_dec(ctxs, ps, &s_k_kk, &c_0, rng)
    }).collect();

    let t0 = Instant::now();
    std::hint::black_box(combine(ctxs, ps, &partials, &c_1, rng));
    let combine = t0.elapsed();

    Timing { setup, encaps, share, combine }
}

fn fmt_d(d: Duration) -> String {
    let us = d.as_secs_f64() * 1_000_000.0;
    if us >= 1_000_000.0 { format!("{:>8.2} s",  us / 1_000_000.0) }
    else if us >= 1000.0 { format!("{:>8.2} ms", us / 1000.0) }
    else                 { format!("{:>8.2} us", us) }
}

fn main() {
    println!("===== Pilvi cost estimate (Cini-Lai-Woo, ePrint 2025/1691) =====");
    println!("Ring: R_q = Z_q[X]/(X^256 + 1)   (f = 512, phi = 256, ell = 1)");
    println!("Pattern: Micciancio-Suhl linear algebra over R_q");
    println!();

    // Build NTT contexts for up to 3 RNS primes. Distinct primes (via the
    // skip parameter) so timings are honest even if values aren't.
    let t0 = Instant::now();
    let ctx_full: Vec<Ctx> = (0..3).map(|i| Ctx::build(D, LOG_D, (i as u64) * 1024)).collect();
    let build_t = t0.elapsed();
    println!("NTT contexts (3 RNS primes, ~2^50 each) built in {}", fmt_d(build_t));
    for (i, c) in ctx_full.iter().enumerate() {
        println!("  prime[{}] = {} (~2^{:.3})", i, c.q, (c.q as f64).log2());
    }
    println!();

    let mut rng = StdRng::from_entropy();

    // ---- Self-test: NTT poly mul agrees with schoolbook on one context.
    // (Schoolbook is too slow to redo per RNS slot.)
    {
        let c = &ctx_full[0];
        let a_std: Vec<u64> = (0..D).map(|_| rng.gen_range(0..c.q)).collect();
        let b_std: Vec<u64> = (0..D).map(|_| rng.gen_range(0..c.q)).collect();
        let a_m: Vec<u64> = a_std.iter().map(|&x| c.to_mont(x)).collect();
        let b_m: Vec<u64> = b_std.iter().map(|&x| c.to_mont(x)).collect();
        let ntt_out_m = c.poly_mul_single(&a_m, &b_m);
        // Schoolbook in standard form:
        let qz = c.q as i128;
        let mut acc = vec![0i128; 2 * D];
        for i in 0..D {
            for j in 0..D { acc[i + j] += a_std[i] as i128 * b_std[j] as i128; }
        }
        let mut sb = vec![0u64; D];
        for i in 0..D {
            let v = (acc[i] - acc[i + D]) % qz;
            sb[i] = if v < 0 { (v + qz) as u64 } else { v as u64 };
        }
        let ntt_out_std: Vec<u64> = ntt_out_m.iter()
            .map(|&x| mont_redc(x as u128, c.q, c.q_neg_inv)).collect();
        assert_eq!(ntt_out_std, sb, "NTT != schoolbook at d = 256");
        println!("Self-test (NTT == schoolbook at d = 256): OK");
        println!();
    }

    // ---- Per-mul calibration for each RNS depth.
    println!("Per-mul calibration:");
    for k in 1..=3 {
        let ctxs = &ctx_full[..k];
        let a = ctxs[0].poly_rand_uniform_mont(&mut rng);
        let b = ctxs[0].poly_rand_uniform_mont(&mut rng);
        const N_CAL: u32 = 1000;
        let t0 = Instant::now();
        for _ in 0..N_CAL { std::hint::black_box(ring_mul(ctxs, &a, &b)); }
        let pmul = t0.elapsed() / N_CAL;
        println!("  k = {} RNS prime(s)  : {} per ring mul (covers log q <= {})",
                 k, fmt_d(pmul), if k == 1 { 50 } else if k == 2 { 85 } else { 130 });
    }
    println!();

    // ---- Main sweep: 8 parameter sets from Table 5.
    println!("=================================================================================================================");
    println!("            Per-phase timings  (Pilvi, Micciancio-Suhl linear algebra over R_q, single-thread)");
    println!("=================================================================================================================");
    println!("{:<24} {:>4} {:>3} {:>3} {:>6} {:>10} {:>10} {:>10} {:>10} {:>11} {:>11}",
             "config", "K", "t", "n", "log_q", "KGen", "Enc", "PdShare", "Combine", "Dec(par)", "Dec(seq)");
    println!("{}", "-".repeat(125));

    for ps in PARAMS {
        let k = rns_primes_for(ps.log_q);
        let ctxs = &ctx_full[..k];
        let tm = run_one(ctxs, ps, &mut rng);
        let dec_par = tm.share + tm.combine;
        let dec_seq = tm.share * (ps.t_thresh as u32) + tm.combine;
        println!("{:<24} {:>4} {:>3} {:>3} {:>6} {:>10} {:>10} {:>10} {:>10} {:>11} {:>11}",
                 ps.name, ps.k_parties, ps.t_thresh, ps.n, ps.q_log_label,
                 fmt_d(tm.setup), fmt_d(tm.encaps),
                 fmt_d(tm.share), fmt_d(tm.combine),
                 fmt_d(dec_par), fmt_d(dec_seq));
    }
    println!();

    // ---- Sizes table for context.
    println!("Sizes (Pilvi Table 5):");
    println!("  {:<24} {:>10} {:>10}  {}", "config", "|ctxt| KB", "|pdk| KB", "regime");
    for ps in PARAMS {
        println!("  {:<24} {:>10.1} {:>10.1}  {}",
                 ps.name, ps.ct_kb, ps.pd_kb, ps.q_descr);
    }
    println!();

    println!("Operation counts (Pilvi Fig. 5, L = 1):");
    println!("  KGen      : 1 vec-mat (n x m ring mults) + 1 K-by-t Vandermonde share");
    println!("                = n*m + K*t*n   ring mults");
    println!("  Enc       : 1 matvec (n x m) + 1 inner product (length m)");
    println!("                = n*m + m       ring mults");
    println!("  ParDec/p  : 1 inner product (length n) + 1 Gaussian ring element");
    println!("                = n             ring mults  (single-round, non-interactive)");
    println!("  Combine   : 1 length-t inner product with subtractive-set coefficients");
    println!("                = t             ring mults");
    println!();
    println!("Legend:");
    println!("  PdShare    = ONE party's ParDec (1 length-n inner product + 1 Gaussian)");
    println!("                 output = 1 ring element (~phi*log q / 8 bytes; 1.7-3.6 KB)");
    println!("  Dec(par)   = PdShare + Combine    (parties act in parallel)");
    println!("  Dec(seq)   = t * PdShare + Combine (sequential simulation on one machine)");
    println!();
    println!("Notes:");
    println!("  * Every Pilvi step is a Micciancio-Suhl style linear-algebra op,");
    println!("    upgraded from u32 scalars to R_q ring scalars. Ring mul = RNS-NTT");
    println!("    at d = 256 with 1, 2, or 3 ~2^50 Proth primes depending on log q.");
    println!("  * ParDec is 0-round and embarrassingly local. Output is ONE ring");
    println!("    element (vs Micciancio-Suhl's one u32 scalar).");
    println!("  * Combine cost is linear in t (the recovery threshold), not in K.");
    println!("    For (t, K) = (2, 8) Combine = 2 ring muls; for (t, K) = (16, 32)");
    println!("    it is 16 ring muls — still cheap compared to KGen's K*t*n.");
    println!("  * V (Vandermonde of the subtractive set) has roots-of-unity entries.");
    println!("    Counted here as full ring mults; a rotation-only path would cut");
    println!("    the K*t*n term substantially.");
    println!("  * Excludes A-from-seed expansion (~ms via SHAKE) and signature");
    println!("    overhead (Pilvi as written is CPA; CCA via [BKW25] adds NIZKs).");
}