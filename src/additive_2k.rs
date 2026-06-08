//! Block 1 — Scalar additive secret sharing over Z_{2^(k+s)} for SPDZ2k.
//!
//! Dishonest-majority additive sharing. Reconstruction is the plain sum (no
//! Lagrange) and needs ALL n parties.
//!
//! The SPDZ2k lift
//! ---------------
//! Secret values for the PKE (secret key, ciphertext coefficients) live in
//! [0, 2^k). We share them in the LIFTED ring Z_{2^(k+s)}, where s is the
//! statistical security parameter (typically s=40). The lift gives statistical
//! hiding: any (n-1) shares are uniform in [0, 2^(k+s)).
//!
//! Notation (matches the paper):
//!   q        = 2^k        (PKE / ciphertext modulus, also called m_base)
//!   q'       = 2^round(log2(q))  ; for q = 2^k this is q' = q (identity switch)
//!   mu       = q / p
//!   mu'      = q' / p     ; equals mu in active branch since q' = q
//!   m_share  = 2^(k+s)    (lifted ring used to share the long-lived secret key)
//!
//! Sharing rule: sample n-1 shares uniformly in [0, 2^(k+s)); the last share
//! makes the sum equal to the secret mod 2^(k+s). Reduce the reconstruction
//! mod 2^k to recover the base value.
//!
//! Storage: u128 (k+s up to ~120 bits with headroom).

use rand::{Rng, RngCore};

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

/// SPDZ2k modulus parameters. Two sharing regimes live here:
///   * `m_share` — the LIFTED ring (2^(k+s)) used to share the long-lived
///     secret key for statistical hiding.
///   * `q, q', mu, mu'` — the BASE rings used in the one-time mask-then-open
///     double sharing. In the active branch q is already a power of 2, so
///     q' = q and mu' = mu; the names follow the paper anyway.
#[derive(Clone, Copy, Debug)]
pub struct SpdzParams {
    pub k: u32,
    pub s: u32,
    pub p: u128,

    pub m_base: u128,     // 2^k         (== q; kept for additive_ring compatibility)
    pub m_share: u128,    // 2^(k+s)     (lifted secret-key sharing modulus)

    pub q: u128,          // 2^k         (ciphertext modulus)
    pub q_prime: u128,    // 2^round(log2(q))    (paper's q'; = q here)
    pub mu: u128,         // q / p       (was 'delta')
    pub mu_prime: u128,   // q' / p      (paper's mu'; = mu here)
}

impl SpdzParams {
    pub fn new(k: u32, s: u32, p: u128) -> Self {
        assert!(p.is_power_of_two(), "p must be a power of two");
        assert!(k + s <= 120, "k+s must fit comfortably in u128");
        assert!(k >= 1, "k must be at least 1");

        let m_base = 1u128 << k;
        let m_share = 1u128 << (k + s);
        let q = m_base;

        // q' = 2^round(log2(q)). For q = 2^k the round() is exact; q' = q.
        let log2_q = (q as f64).log2();
        let k_prime = log2_q.round() as u32;
        let q_prime: u128 = 1u128 << k_prime;
        assert!(q_prime <= q, "q' must be <= q for mod switching down");
        assert!(q_prime % p == 0, "p must divide q'");

        SpdzParams {
            k,
            s,
            p,
            m_base,
            m_share,
            q,
            q_prime,
            mu: q / p,
            mu_prime: q_prime / p,
        }
    }
}

// ---------------------------------------------------------------------------
// A scalar additive share
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AddShare {
    pub x: u32,
    pub y: u128,
}

// ---------------------------------------------------------------------------
// Share / reconstruct over an arbitrary 2^X modulus
// ---------------------------------------------------------------------------

pub fn share(value: u128, n: usize, modulus: u128) -> Vec<AddShare> {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::from_entropy();
    share_with_rng(value, n, modulus, &mut rng)
}

pub fn share_with_rng<R: RngCore>(
    value: u128,
    n: usize,
    modulus: u128,
    rng: &mut R,
) -> Vec<AddShare> {
    assert!(n >= 2, "additive sharing needs at least 2 parties");
    debug_assert!(value < modulus, "value must be in [0, modulus)");

    use rand::Rng as _;

    let mut shares = Vec::with_capacity(n);
    let mut acc: u128 = 0;
    for i in 0..(n - 1) {
        let v = rng.gen_range(0..modulus);
        shares.push(AddShare { x: (i + 1) as u32, y: v });
        acc = (acc + v) % modulus;
    }
    let last = if value >= acc { value - acc } else { modulus + value - acc };
    shares.push(AddShare { x: n as u32, y: last });

    debug_assert_eq!(
        shares.iter().fold(0u128, |a, s| (a + s.y) % modulus),
        value,
        "share-sum invariant"
    );
    shares
}

pub fn reconstruct(shares: &[AddShare], modulus: u128) -> u128 {
    shares.iter().fold(0u128, |acc, s| (acc + s.y) % modulus)
}

// ---------------------------------------------------------------------------
// Lifted sharing: secret in [0, 2^k) shared over Z_{2^(k+s)}
// ---------------------------------------------------------------------------

pub fn share_lifted(secret_mod_2k: u128, n: usize, params: &SpdzParams) -> Vec<AddShare> {
    debug_assert!(
        secret_mod_2k < params.m_base,
        "secret must be in [0, 2^k); got {secret_mod_2k} >= {}",
        params.m_base
    );
    share(secret_mod_2k, n, params.m_share)
}

pub fn reconstruct_lifted(shares: &[AddShare], params: &SpdzParams) -> u128 {
    let lifted = reconstruct(shares, params.m_share);
    lifted % params.m_base
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_share_reconstruct_roundtrip_basic() {
        let modulus = 1u128 << 60;
        let v = 123_456_789_u128;
        let shares = share(v, 5, modulus);
        assert_eq!(shares.len(), 5);
        assert_eq!(reconstruct(&shares, modulus), v);
    }

    #[test]
    fn test_lifted_share_recovers_base() {
        let p = SpdzParams::new(20, 40, 2);
        let secret: u128 = 0xABCDE;
        let shares = share_lifted(secret, 4, &p);
        assert_eq!(reconstruct_lifted(&shares, &p), secret);
    }

    #[test]
    fn test_paper_notation_consistency() {
        // q' = q, mu' = mu in the active branch (q is already a power of 2).
        let p = SpdzParams::new(30, 40, 2);
        assert_eq!(p.q, p.q_prime);
        assert_eq!(p.mu, p.mu_prime);
        assert_eq!(p.mu, p.q / 2);
    }
}