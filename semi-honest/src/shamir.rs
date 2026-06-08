//! Block 1 — Shamir secret sharing primitives over the prime field F_q.
//!
//! q = 8380417 is prime, so F_q is a field and Shamir secret sharing works
//! directly (no Galois-ring machinery needed). Everything here is SCALAR
//! field arithmetic on u32 values reduced mod q; the Ring/Vector layer that
//! shares the actual secret key is Block 2 and builds on these functions.
//!
//! Conventions
//! -----------
//! * Field elements are u32 in [0, q). Products are done in u64 then reduced,
//!   since a product of two ~23-bit values is ~46 bits.
//! * Party j (j in 1..=n) is assigned the nonzero evaluation point x_j = j.
//!   Shamir requires distinct nonzero points; j in 1..=n satisfies that as
//!   long as n < q (always true here).
//! * A degree-t polynomial f with f(0) = secret is stored as its coefficient
//!   list [a_0, a_1, ..., a_t] with a_0 = secret (so f(x) = sum a_i x^i).

use crate::params;

/// Field addition mod q.
#[inline]
pub fn add(a: u32, b: u32) -> u32 {
    ((a as u64 + b as u64) % params::Q64) as u32
}

/// Field subtraction mod q.
#[inline]
pub fn sub(a: u32, b: u32) -> u32 {
    // a + q - b stays < 2q < 2^24, safe even before reduction, but reduce anyway.
    ((a as u64 + params::Q64 - b as u64) % params::Q64) as u32
}

/// Field multiplication mod q.
#[inline]
pub fn mul(a: u32, b: u32) -> u32 {
    ((a as u64 * b as u64) % params::Q64) as u32
}

/// Modular inverse via Fermat's little theorem: a^(q-2) mod q.
/// Valid because q is prime and a != 0. Panics on a == 0.
///
/// NOTE: `fastmodpow` takes an exponent of type u8, but q-2 is ~23 bits, so
/// we can't call it directly. We do our own square-and-multiply in u64 here.
pub fn inverse(a: u32) -> u32 {
    assert!(a % params::Q != 0, "no inverse for 0 in F_q");
    mod_pow(a, params::Q64 - 2)
}

/// Square-and-multiply for an arbitrary u64 exponent, mod q.
/// (Standalone because util::fastmodpow caps the exponent at u8.)
pub fn mod_pow(base: u32, exp: u64) -> u32 {
    let mut base = base as u64 % params::Q64;
    let mut exp = exp;
    let mut result: u64 = 1;
    while exp > 0 {
        if exp & 1 == 1 {
            result = (result * base) % params::Q64;
        }
        exp >>= 1;
        base = (base * base) % params::Q64;
    }
    result as u32
}

/// Evaluate f(x) = sum_i coeffs[i] * x^i  over F_q (Horner's method).
/// coeffs[0] is the constant term (the secret when used for sharing).
pub fn eval_poly_at(coeffs: &[u32], x: u32) -> u32 {
    let mut acc: u32 = 0;
    // Horner: start from the highest-degree coefficient.
    for &c in coeffs.iter().rev() {
        acc = add(mul(acc, x), c);
    }
    acc
}

/// Lagrange coefficient lambda_j for reconstructing f(0) from the shares held
/// by the parties whose evaluation points are `active_ids` (the t+1 active
/// parties). `j` is the evaluation point of the party whose coefficient we
/// want. This is the "local Lagrange" multiplier used to convert a degree-t
/// Shamir share into an additive share:
///
///     lambda_j = prod_{m in active, m != j}  m / (m - j)   (all mod q)
///
/// Then  secret = sum_{j in active} lambda_j * share_j.
pub fn lagrange_coeff_at_zero(active_ids: &[u32], j: u32) -> u32 {
    let mut num: u32 = 1; // product of m
    let mut den: u32 = 1; // product of (m - j)
    for &m in active_ids {
        if m == j {
            continue;
        }
        num = mul(num, m);
        den = mul(den, sub(m, j)); // (m - j) mod q
    }
    mul(num, inverse(den))
}

/// A single Shamir share: the secret-sharing value f(x) together with the
/// evaluation point x it was produced at. Carrying x avoids any implicit
/// index->point coupling and is exactly what reconstruction / Lagrange need.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Share {
    pub x: u32,     // evaluation point (party id), nonzero
    pub y: u32,     // f(x), the share value in F_q
}

/// Sample one uniform field element in [0, q) from a provided RNG, via
/// rejection sampling on 4 random bytes. Rejection keeps the distribution
/// unbiased (a plain `u32 % q` would slightly favor small values).
///
/// Takes the RNG by reference so callers can seed ONCE and draw many elements;
/// seeding StdRng from entropy per draw (the old behavior) was the dominant
/// cost when sharing a 256-coefficient Ring.
pub fn random_field_element<R: rand::RngCore>(rng: &mut R) -> u32 {
    loop {
        let candidate = rng.next_u32();
        if candidate < params::Q {
            return candidate;
        }
        // reject and redraw; acceptance prob ~ q/2^32
    }
}

/// Shamir Share(x): produce an (t, n) degree-t sharing of `secret` over F_q.
///
/// Convenience wrapper that seeds one StdRng from entropy and delegates to
/// `share_with_rng`. For bulk sharing (e.g. 256 coefficients of a Ring),
/// prefer seeding a single RNG yourself and calling `share_with_rng` in a loop
/// to avoid repeated entropy seeding.
pub fn share(secret: u32, t: usize, n: usize) -> Vec<Share> {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::from_entropy();
    share_with_rng(secret, t, n, &mut rng)
}

/// Shamir Share(x) using a caller-provided RNG.
///
/// Samples f(X) = secret + a_1 X + ... + a_t X^t with uniform a_1..a_t, then
/// evaluates at points x = 1, 2, ..., n. Any t+1 of the returned shares
/// reconstruct the secret; any t reveal nothing.
///
/// Requirements: t < n, and n < q (so eval points are distinct nonzero).
pub fn share_with_rng<R: rand::RngCore>(
    secret: u32,
    t: usize,
    n: usize,
    rng: &mut R,
) -> Vec<Share> {
    assert!(t < n, "need t < n for a (t,n) sharing");
    assert!((n as u64) < params::Q64, "n must be < q for distinct points");

    // Build polynomial coefficients [secret, a_1, ..., a_t].
    let mut coeffs = Vec::with_capacity(t + 1);
    coeffs.push(secret % params::Q);
    for _ in 0..t {
        coeffs.push(random_field_element(rng));
    }

    // Evaluate at x = 1..=n.
    (1..=n as u32)
        .map(|x| Share { x, y: eval_poly_at(&coeffs, x) })
        .collect()
}

/// Reconstruct the secret f(0) from any t+1 (or more) shares using Lagrange
/// interpolation at zero. The caller must pass at least t+1 shares with
/// distinct x; passing more is fine (the first that reconstruct consistently
/// determine f(0)).
pub fn reconstruct(shares: &[Share]) -> u32 {
    let active_ids: Vec<u32> = shares.iter().map(|s| s.x).collect();
    let mut secret: u32 = 0;
    for s in shares {
        let lambda = lagrange_coeff_at_zero(&active_ids, s.x);
        secret = add(secret, mul(lambda, s.y));
    }
    secret
}
