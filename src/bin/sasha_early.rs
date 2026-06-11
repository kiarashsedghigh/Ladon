//! Cost estimator for **sasha1** = Lapiha & Prest, "A Lattice-Based IND-CCA
//! Threshold KEM from the BCHK+ Transform", Asiacrypt 2025 / ePrint 2025/1958.
//!
//! Sweeps N ∈ {4, 8, 16, 32} parties at the single Table-2 parameter set:
//!   κ = 128, T = 32, Q_Dec = 2^40, d = 4096, q ≈ 2^100, β = 2^77,
//!   ς_a = 8, ς = ς_RLWE = 4, ς_p = 2^47, |ek| = 50 KiB, |ct| = 450 KiB.
//!
//! Per-phase operation counts (paper Algorithms 1, 4, 5-7, 8):
//!   TIBE.Setup       Alg. 1   : 6 poly muls + 2 Gaussian polys + (T,N) share
//!   TIBE.Encrypt     Alg. 4   : 12 poly muls + 11 Gaussian polys
//!     (2 for E(vk)·G; 9 for u = F_vk^T · s + e ; 1 for v = r·s + e' + msg)
//!   ShareExtract / p Algs.5-7 : 12 poly muls + 9 Gaussian polys
//!     (Round 0: 1 for a0·p; 3 for (A1−E(vk)G)·x0; 3 for A2·x1; 2 for E(vk)·G;
//!      Round 2: 1 for d0^{-1}·(r−w); 2 for c0·λ·share)
//!   Combine          Alg. 8   : 18 poly muls (9 for F_vk·z assert, 9 for v−u·z)
//!
//! Implementation strategy: 3-prime RNS at q ≈ 2^100
//! --------------------------------------------------
//! q ≈ 2^100 doesn't fit in u64. The standard way to NTT at this size is
//! Residue Number System: pick k NTT-friendly primes q_1, ..., q_k with
//! q_1·...·q_k ≥ q, represent each ring coefficient as a tuple
//! (x mod q_1, ..., x mod q_k), and do k independent NTTs per polynomial
//! multiplication. Each NTT is over a ~2^50 prime — exactly the regime our
//! Montgomery NTT was tuned for in the sasha2 estimator.
//!
//! The choice of k matters:
//!   * k = 2 with 50-bit primes gives a product of EXACTLY 2^100, leaving
//!     no headroom for intermediate convolution sums (which transiently
//!     reach ~d·q^2 / q ~ d·q before reduction) — unsafe in practice.
//!   * k = 3 with ~50-bit primes gives ~2^150, which comfortably covers
//!     intermediate values and matches what CKKS/BFV implementations use.
//!   * k = 4+ adds further margin but no functional benefit for q = 2^100.
//!
//! We use k = 3, so every "ring multiplication" in sasha1 is 3 independent
//! single-prime NTT muls. The third polynomial result is `black_box`'d to
//! prevent LTO from dead-code-eliminating it.
//!
//! For *timing* this estimator doesn't need RNS-correct arithmetic, only
//! its true cost. We build a single Montgomery NTT context at d = 4096
//! with q ≈ 2^50 and define `poly_mul_q100` as three back-to-back calls
//! to the single-prime poly mul. This faithfully reflects 3-prime RNS:
//!   * 3 × 2 = 6 forward NTTs per multiplication
//!   * 3 pointwise mul rounds
//!   * 3 inverse NTTs
//! plus one CRT reconstruction per polynomial (~3·D u128 ops ≈ ~60 µs at
//! d=4096, included as a black_box at the end).
//!
//! Build instructions
//! ------------------
//!   Drop into examples/bench_sasha1.rs (or src/bin/), then:
//!     cargo run --release --example bench_sasha1
//!
//! Compared to sasha2
//! ------------------
//! sasha1 vs the comparable sasha2 κ=128 non-robust row:
//!   * 9× larger ciphertext (450 KiB vs 28 KiB)
//!   * larger poly mul (2× via RNS) AND more muls per phase (12 vs 4, 18 vs 3)
//!   * but no per-share verify in Combine (non-robust), so Combine is flat in N
//!   * net: expect ~10× slower than sasha2 κ=128 non-robust at the same N

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::time::{Duration, Instant};

// =========================================================================
// sasha1 parameter set (Table 2 of the paper).
// =========================================================================

const D: usize          = 4096;
const LOG_D: u32        = 12;
const TARGET_LOG_Q: u32 = 50;             // per RNS prime; q1·q2 ≥ 2^100
const SIGMA_A: f64      = 8.0;
const SIGMA: f64        = 4.0;            // = sigma_RLWE
const SIGMA_P: f64      = 140_737_488_355_328.0;  // 2^47
const KAPPA: usize      = 128;
const T_THRESHOLD: usize = 32;            // threshold
const EK_BYTES: usize   = 50 * 1024;
const CT_BYTES: usize   = 450 * 1024;
const PARTY_COUNTS: &[usize] = &[4, 8, 16, 32];

// =========================================================================
// Montgomery arithmetic over a ~50-bit prime (one RNS component).
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
// NTT context for one RNS component (q ≈ 2^50).
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
    fn build(d: usize, log_d: u32) -> Self {
        let two_d = 2 * d as u64;
        let lo = 1u64 << (TARGET_LOG_Q - 1);
        let mut k = (lo + two_d - 1) / two_d;
        let mut tried: u64 = 0;
        let q = loop {
            let cand = k * two_d + 1;
            if is_prime(cand) { break cand; }
            k += 1;
            tried += 1;
            if tried > 1_000_000 { panic!("Proth search exhausted"); }
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

    /// Single-prime (~2^50) negacyclic poly mul. Building block for the
    /// 2-prime RNS routine `poly_mul_q100` below.
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
// 3-prime RNS poly mul: 3× the cost of a single-prime NTT mul, plus a CRT
// reconstruction step. Used for every ring multiplication in sasha1.
// =========================================================================

/// Number of RNS primes. With q_i ~ 2^50 each, k = 3 gives q_1·q_2·q_3 ~ 2^150,
/// comfortably covering q = 2^100 plus headroom for intermediate convolution
/// sums during a multiplication (which transiently reach d·q^2/q ~ d·q ~ 2^112).
/// Standard choice in CKKS/BFV implementations at this modulus size.
const RNS_PRIMES: usize = 3;

fn poly_mul_q100(c: &Ctx, a: &[u64], b: &[u64]) -> Vec<u64> {
    // Component 1: NTT mul mod q_1.
    let r1 = c.poly_mul_single(a, b);

    // Components 2..k: real RNS uses distinct Proth primes with their own
    // twiddle tables; the cost per NTT is the same regardless. We reuse the
    // single context — meaningless values, identical timing. `black_box`
    // prevents LTO from dead-code-eliminating these calls.
    for _ in 1..RNS_PRIMES {
        let r_i = c.poly_mul_single(a, b);
        std::hint::black_box(&r_i);
    }

    // CRT reconstruction: rebuild each coefficient in Z_q from its k RNS
    // components via one big-int mul + add per coefficient. Cost is
    // ~k * D u128 ops ≈ ~60 µs at d=4096, k=3. We model it with a small
    // explicit loop so it actually appears in the timing.
    let mut crt_sink: u64 = 0;
    for i in 0..c.d {
        for _ in 0..RNS_PRIMES {
            crt_sink = crt_sink.wrapping_add(mont_mul(r1[i], c.r2_mod_q, c.q, c.q_neg_inv));
        }
    }
    std::hint::black_box(crt_sink);

    r1
}

// =========================================================================
// sasha1 protocol phases. Counts follow Algorithms 1, 4, 5-7, 8.
// =========================================================================

struct Setup {
    a0_d0: Vec<u64>,    // = a0 · d0
    b0_d0: Vec<u64>,    // = b0 · d0
    a1: [Vec<u64>; 3],
    a2_entries: [Vec<u64>; 2], // a2·d2, b2·d2  (the "1·d2" entry is just d2)
    g_powers: [Vec<u64>; 2],   // g, g^2
    r: Vec<u64>,
}

fn tibe_setup(c: &Ctx, n_parties: usize, rng: &mut StdRng) -> Setup {
    // Alg. 1 lines 1-3: sample d0, a0; sa, ea; compute b0 = a0·sa + ea − β;
    //                   then A0 = [1, a0, b0]·d0.
    let d0 = c.poly_rand_uniform_mont(rng);
    let a0 = c.poly_rand_uniform_mont(rng);
    let sa = c.sample_poly_gauss_mont(SIGMA_A, rng);
    let _ea = c.sample_poly_gauss_mont(SIGMA_A, rng);
    let a0_sa = poly_mul_q100(c, &a0, &sa);                    // 1 mul
    // b0 := a0·sa + ea − β  is just a poly_add then a const sub; treat as free.
    let b0 = a0_sa;
    let a0_d0 = poly_mul_q100(c, &a0, &d0);                    // 1 mul
    let b0_d0 = poly_mul_q100(c, &b0, &d0);                    // 1 mul

    // Alg. 1 lines 4-5: A1 ← R^3_q ;  A2 = [1, a2, b2]·d2.
    let a1 = [
        c.poly_rand_uniform_mont(rng),
        c.poly_rand_uniform_mont(rng),
        c.poly_rand_uniform_mont(rng),
    ];
    let d2 = c.poly_rand_uniform_mont(rng);
    let a2 = c.poly_rand_uniform_mont(rng);
    let b2 = c.poly_rand_uniform_mont(rng);
    let a2_d2 = poly_mul_q100(c, &a2, &d2);                    // 1 mul
    let b2_d2 = poly_mul_q100(c, &b2, &d2);                    // 1 mul

    // Alg. 1 line 6: G = [1, g, g^2].
    let g = c.poly_rand_uniform_mont(rng);
    let g2 = poly_mul_q100(c, &g, &g);                         // 1 mul

    // Alg. 1 line 7: r ← R_q.
    let r = c.poly_rand_uniform_mont(rng);

    // Alg. 1 line 9: Shamir (T, N) share of (sa, ea). Approximated by
    // 2·d·N scalar mults — peanuts compared to the six poly muls above
    // but counted for completeness.
    let mut sink: u64 = 0;
    for _ in 0..(2 * c.d * n_parties) {
        sink = sink.wrapping_add(c.mm(rng.gen_range(0..c.q), rng.gen_range(0..c.q)));
    }
    std::hint::black_box(sink);

    Setup {
        a0_d0, b0_d0,
        a1, a2_entries: [a2_d2, b2_d2],
        g_powers: [g, g2],
        r,
    }
}

fn tibe_encrypt(c: &Ctx, setup: &Setup, rng: &mut StdRng) {
    // Alg. 4 line 1: F_vk = [A0 | A1 − E(vk)·G | A2].
    // E(vk)·G = [E(vk), E(vk)·g, E(vk)·g^2].
    let e_vk = c.poly_rand_uniform_mont(rng);     // H(id) modelled as URS
    let _eg1 = poly_mul_q100(c, &e_vk, &setup.g_powers[0]);    // E(vk)·g     (1)
    let _eg2 = poly_mul_q100(c, &e_vk, &setup.g_powers[1]);    // E(vk)·g^2   (2)

    // Alg. 4 lines 2-4: sample s, e, e'.
    let s   = c.sample_poly_gauss_mont(SIGMA, rng);             // 1 Gaussian
    let _e0 = c.sample_poly_gauss_mont(SIGMA, rng);             // 3 + 3 + 3 = 9 Gaussians
    let _e1 = c.sample_poly_gauss_mont(SIGMA, rng);
    let _e2 = c.sample_poly_gauss_mont(SIGMA, rng);
    let _e3 = c.sample_poly_gauss_mont(SIGMA, rng);
    let _e4 = c.sample_poly_gauss_mont(SIGMA, rng);
    let _e5 = c.sample_poly_gauss_mont(SIGMA, rng);
    let _e6 = c.sample_poly_gauss_mont(SIGMA, rng);
    let _e7 = c.sample_poly_gauss_mont(SIGMA, rng);
    let _e8 = c.sample_poly_gauss_mont(SIGMA, rng);
    let _ep = c.sample_poly_gauss_mont(SIGMA, rng);             // e'

    // Alg. 4 line 5: u = F_vk^T · s + e.  F_vk has 9 ring entries, s is a
    // single ring element, so 9 poly muls.  We reuse setup.a0_d0 / b0_d0 etc.
    let _u0 = poly_mul_q100(c, &setup.a0_d0,           &s);    // (3)
    let _u1 = poly_mul_q100(c, &setup.b0_d0,           &s);    // (4)
    let _u2 = poly_mul_q100(c, &c.poly_rand_uniform_mont(rng), &s); // 1·d0 entry (5)
    let _u3 = poly_mul_q100(c, &setup.a1[0],           &s);    // (6)
    let _u4 = poly_mul_q100(c, &setup.a1[1],           &s);    // (7)
    let _u5 = poly_mul_q100(c, &setup.a1[2],           &s);    // (8)
    let _u6 = poly_mul_q100(c, &setup.a2_entries[0],   &s);    // (9)
    let _u7 = poly_mul_q100(c, &setup.a2_entries[1],   &s);    // (10)
    let _u8 = poly_mul_q100(c, &c.poly_rand_uniform_mont(rng), &s); // 1·d2 entry (11)

    // Alg. 4 line 6: v = r·s + e' + Encode(msg).
    let _v  = poly_mul_q100(c, &setup.r, &s);                  // (12)
}

fn share_extract_one(c: &Ctx, setup: &Setup, rng: &mut StdRng) {
    // E(vk)·G — could be cached across parties; counted here so per-party
    // cost is upper-bounded.
    let e_vk = c.poly_rand_uniform_mont(rng);
    let _eg1 = poly_mul_q100(c, &e_vk, &setup.g_powers[0]);    // (1)
    let _eg2 = poly_mul_q100(c, &e_vk, &setup.g_powers[1]);    // (2)

    // Alg. 5 (Round 0): sample p_i, x_{i,0}, x_{i,1} (9 Gaussians);
    //                    compute w_i := y_0 + y_1 + y_2.
    let p_i0 = c.sample_poly_gauss_mont(SIGMA_P, rng);
    let p_i1 = c.sample_poly_gauss_mont(SIGMA_P, rng);
    let _p_i2 = c.sample_poly_gauss_mont(SIGMA_P, rng);
    let x_00 = c.sample_poly_gauss_mont(SIGMA_P, rng);
    let x_01 = c.sample_poly_gauss_mont(SIGMA_P, rng);
    let x_02 = c.sample_poly_gauss_mont(SIGMA_P, rng);
    let x_10 = c.sample_poly_gauss_mont(SIGMA_P, rng);
    let x_11 = c.sample_poly_gauss_mont(SIGMA_P, rng);
    let x_12 = c.sample_poly_gauss_mont(SIGMA_P, rng);

    let a0 = &setup.a0_d0; // proxy for a0 (timing-equivalent)
    let _y0 = poly_mul_q100(c, a0, &p_i0);                     // 1 mul: a0·p_{i,0}
    // y_1 = (A1 − E(vk)G) · x_{i,0} — 3 muls
    let _y10 = poly_mul_q100(c, &setup.a1[0], &x_00);          // (4)
    let _y11 = poly_mul_q100(c, &setup.a1[1], &x_01);          // (5)
    let _y12 = poly_mul_q100(c, &setup.a1[2], &x_02);          // (6)
    // y_2 = A2 · x_{i,1} — 3 muls
    let _y20 = poly_mul_q100(c, &setup.a2_entries[0], &x_10);  // (7)
    let _y21 = poly_mul_q100(c, &setup.a2_entries[1], &x_11);  // (8)
    let _y22 = poly_mul_q100(c, &c.poly_rand_uniform_mont(rng), &x_12); // (9)
    // (Hcmt(w_i) hash: tens of us, ignore.)

    // Alg. 7 (Round 2): compute (c0, c1) := Decomp_β(d_0^{-1} · (r − w)),
    //                   then z_{i,0} = p_{i,0} + c0·λ·⟦ea⟧_i + m_{i,0}, etc.
    let d0_inv = c.poly_rand_uniform_mont(rng);
    let r_minus_w = c.poly_add(&setup.r, &p_i1); // placeholder sub→add
    let c0_c1 = poly_mul_q100(c, &d0_inv, &r_minus_w);         // (10)
    // z_{i,0}: c0 · (λ ⟦ea⟧_i)   (scalar-poly mul absorbed; full poly mul counted)
    let ea_share = c.sample_poly_gauss_mont(SIGMA_A, rng);
    let _z_i0 = poly_mul_q100(c, &c0_c1, &ea_share);           // (11)
    // z_{i,1}: c0 · (λ ⟦sa⟧_i)
    let sa_share = c.sample_poly_gauss_mont(SIGMA_A, rng);
    let _z_i1 = poly_mul_q100(c, &c0_c1, &sa_share);           // (12)
}

fn combine_one(c: &Ctx, setup: &Setup, _n_parties: usize, rng: &mut StdRng) {
    // Alg. 8 line 7: assert F_vk · [z, x0, x1]^T = r.
    // 9-vector inner product = 9 poly muls.
    let z = [
        c.poly_rand_uniform_mont(rng),
        c.poly_rand_uniform_mont(rng),
        c.poly_rand_uniform_mont(rng),
    ];
    let _x0 = c.poly_rand_uniform_mont(rng);
    let _x1 = c.poly_rand_uniform_mont(rng);
    // F_vk has 9 entries: A0 (3) | A1 - E(vk)G (3) | A2 (3).
    let _va0 = poly_mul_q100(c, &setup.a0_d0, &z[0]);          // 1
    let _va1 = poly_mul_q100(c, &setup.b0_d0, &z[1]);          // 2
    let _va2 = poly_mul_q100(c, &c.poly_rand_uniform_mont(rng), &z[2]); // 3
    let _vb0 = poly_mul_q100(c, &setup.a1[0], &z[0]);          // 4
    let _vb1 = poly_mul_q100(c, &setup.a1[1], &z[1]);          // 5
    let _vb2 = poly_mul_q100(c, &setup.a1[2], &z[2]);          // 6
    let _vc0 = poly_mul_q100(c, &setup.a2_entries[0], &z[0]);  // 7
    let _vc1 = poly_mul_q100(c, &setup.a2_entries[1], &z[1]);  // 8
    let _vc2 = poly_mul_q100(c, &c.poly_rand_uniform_mont(rng), &z[2]); // 9

    // Alg. 8 line 8: msg := Decode(v − [z, x0, x1]^T · u).
    // Another 9-vector inner product = 9 poly muls.
    let u_polys: Vec<Vec<u64>> = (0..9).map(|_| c.poly_rand_uniform_mont(rng)).collect();
    let z_full: Vec<Vec<u64>>  = (0..9).map(|_| c.poly_rand_uniform_mont(rng)).collect();
    for i in 0..9 {
        let _ = poly_mul_q100(c, &z_full[i], &u_polys[i]);     // 10..18
    }
}

// =========================================================================
// Driver
// =========================================================================

#[derive(Clone, Copy)]
struct Timing { setup: Duration, encaps: Duration, share: Duration, combine: Duration }

fn run_one(c: &Ctx, n_parties: usize, rng: &mut StdRng) -> Timing {
    let t0 = Instant::now();
    let setup_ = tibe_setup(c, n_parties, rng);
    let setup = t0.elapsed();
    let t0 = Instant::now();
    tibe_encrypt(c, &setup_, rng);
    let encaps = t0.elapsed();
    let t0 = Instant::now();
    share_extract_one(c, &setup_, rng);
    let share = t0.elapsed();
    let t0 = Instant::now();
    combine_one(c, &setup_, n_parties, rng);
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
    println!("===== sasha1 cost estimate (Lapiha-Prest, Asiacrypt'25) =====");
    println!("Parameters: kappa = {}, d = {}, q ≈ 2^100 ({}-prime RNS), T = {}",
             KAPPA, D, RNS_PRIMES, T_THRESHOLD);
    println!("Sizes: |ek| = {} KiB  |ct| = {} KiB",
             EK_BYTES / 1024, CT_BYTES / 1024);
    println!();

    let t0 = Instant::now();
    let c = Ctx::build(D, LOG_D);
    let build_t = t0.elapsed();
    println!("NTT context (single RNS prime, ~2^50) built in {}", fmt_d(build_t));
    println!("  q_RNS = {} (~2^{:.3})", c.q, (c.q as f64).log2());
    println!();

    let mut rng = StdRng::from_entropy();

    // Calibrate single-prime mul, then 2-prime RNS mul.
    let a = c.poly_rand_uniform_mont(&mut rng);
    let b = c.poly_rand_uniform_mont(&mut rng);
    const N_CAL: u32 = 30;
    let t0 = Instant::now();
    for _ in 0..N_CAL { std::hint::black_box(c.poly_mul_single(&a, &b)); }
    let pmul1 = t0.elapsed() / N_CAL;
    let t0 = Instant::now();
    for _ in 0..N_CAL { std::hint::black_box(poly_mul_q100(&c, &a, &b)); }
    let pmul2 = t0.elapsed() / N_CAL;
    println!("Per-mul calibration at d = {}:", D);
    println!("  single-prime NTT (~2^50 modulus)         : {}", fmt_d(pmul1));
    println!("  {}-prime RNS mul (~2^100 effective + CRT) : {}", RNS_PRIMES, fmt_d(pmul2));
    println!();

    println!("================================================================");
    println!("  Per-phase timings (Montgomery NTT, 2-prime RNS, single-thread)");
    println!("================================================================");
    println!("{:>3} {:>10} {:>10} {:>10} {:>10} {:>11} {:>11}",
             "N", "Setup", "Encaps", "Share/p", "Combine", "Decap(seq)", "Decap(par)");
    println!("{}", "-".repeat(74));

    for &n in PARTY_COUNTS {
        let t = run_one(&c, n, &mut rng);
        let total_seq = t.share * n as u32 + t.combine;
        let total_par = t.share + t.combine;
        println!("{:>3} {:>10} {:>10} {:>10} {:>10} {:>11} {:>11}",
                 n, fmt_d(t.setup), fmt_d(t.encaps),
                 fmt_d(t.share), fmt_d(t.combine),
                 fmt_d(total_seq), fmt_d(total_par));
    }

    println!();
    println!("Operation counts (paper algorithms 1, 4, 5-7, 8):");
    println!("  Setup       : 6 poly muls  (a0·sa, a0·d0, b0·d0, a2·d2, b2·d2, g·g)");
    println!("  Encaps      : 12 poly muls (2 for E(vk)·G + 9 for u + 1 for v)");
    println!("  ShareExtract: 12 poly muls (2 + 1 + 3 + 3 in R0; 1 + 2 in R2)");
    println!("  Combine     : 18 poly muls (9 for F_vk·z assert + 9 for v − u·z)");
    println!();
    println!("Legend:");
    println!("  Share/p     = one party's ShareExtract (rounds 0 + 1 + 2)");
    println!("  Decap(seq)  = N · Share/p + Combine  (sequential committee)");
    println!("  Decap(par)  = max Share/p + Combine  (parallel committee, then TEE)");
    println!();
    println!("Notes:");
    println!("  * q ≈ 2^100 emulated via {}-prime RNS: every poly mul = {}× NTT mul + CRT",
             RNS_PRIMES, RNS_PRIMES);
    println!("  * sasha1 is non-robust → Combine has no per-share verify (flat in N)");
    println!("  * vs sasha2 κ=128 non-robust: ~10x more poly muls per phase + 2x per mul");
    println!("  * excludes WOTS+ signing (~ms range, hash-heavy)");
}