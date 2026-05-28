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



#[test]
fn test_inverse_roundtrip() {
    for a in [1u32, 2, 17, 1753, 123456, params::Q - 1] {
        assert_eq!(mul(a, inverse(a)), 1, "a * a^-1 should be 1 for a={a}");
    }
}

#[test]
fn test_eval_constant_poly() {
    // f(x) = 42 (degree 0): evaluates to 42 everywhere.
    let f = [42u32];
    assert_eq!(eval_poly_at(&f, 0), 42);
    assert_eq!(eval_poly_at(&f, 5), 42);
}

#[test]
fn test_eval_linear_poly() {
    // f(x) = 3 + 5x  =>  f(0)=3, f(1)=8, f(2)=13
    let f = [3u32, 5u32];
    assert_eq!(eval_poly_at(&f, 0), 3);
    assert_eq!(eval_poly_at(&f, 1), 8);
    assert_eq!(eval_poly_at(&f, 2), 13);
}

#[test]
fn test_lagrange_reconstructs_secret() {
    // Degree-1 sharing of secret=100 with poly f(x) = 100 + 7x.
    // Shares at x=1,2,3. Any 2 of them must reconstruct 100 via local
    // Lagrange (sum of lambda_j * share_j).
    let secret = 100u32;
    let f = [secret, 7u32];
    let ids = [1u32, 2u32, 3u32];
    let shares: Vec<u32> = ids.iter().map(|&x| eval_poly_at(&f, x)).collect();

    // Use the first t+1 = 2 parties (ids 1 and 2).
    let active = [1u32, 2u32];
    let mut recon: u32 = 0;
    for (idx, &id) in active.iter().enumerate() {
        let lambda = lagrange_coeff_at_zero(&active, id);
        recon = add(recon, mul(lambda, shares[idx]));
    }
    assert_eq!(recon, secret);

    // A different active set (ids 2 and 3) must also work.
    let active2 = [2u32, 3u32];
    let mut recon2: u32 = 0;
    for &id in active2.iter() {
        let share = shares[(id - 1) as usize];
        let lambda = lagrange_coeff_at_zero(&active2, id);
        recon2 = add(recon2, mul(lambda, share));
    }
    assert_eq!(recon2, secret);
}

#[test]
fn test_share_reconstruct_roundtrip() {
    // Real (t,n) sharing via the public API, with random polynomial.
    let secret = 1234567u32 % params::Q;
    let t = 2;
    let n = 5;
    let shares = share(secret, t, n);
    assert_eq!(shares.len(), n);

    // Any t+1 = 3 shares reconstruct; t = 2 generally do not.
    let subset: Vec<Share> = shares[0..t + 1].to_vec();
    assert_eq!(reconstruct(&subset), secret);

    // A different t+1 subset also reconstructs.
    let subset2: Vec<Share> = shares[2..5].to_vec();
    assert_eq!(reconstruct(&subset2), secret);

    // Using all n shares (over-determined) still gives the secret.
    assert_eq!(reconstruct(&shares), secret);
}
