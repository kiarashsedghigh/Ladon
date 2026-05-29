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
//! Sharing rule: sample n-1 shares uniformly in [0, 2^(k+s)); the last share
//! makes the sum equal to the secret mod 2^(k+s). Reduce the reconstruction
//! mod 2^k to recover the base value.
//!
//! Storage: u128 (k+s up to ~120 bits with headroom). Operations done in u128;
//! sums never overflow because both addends are < 2^(k+s) < 2^120.
//!
//! Conventions
//! -----------
//! Party indices are i in 1..=n (matching the shamir layer for consistency, so
//! "party j" is the same identifier across both branches). Share i is the
//! value held by party i.

use rand::{Rng, RngCore};

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

/// SPDZ2k modulus parameters. Two distinct sharing rings live here:
///
///   * **m_share / m_phi**  — the LIFTED rings (2^(k+s) and 2^(k+s)/p) used
///     for sharing the long-lived secret key. The lift gives the statistical
///     hiding that distinguishes SPDZ2k from naive additive sharing.
///   * **q / delta**        — the BASE rings (2^k and 2^(k-1)) used for the
///     one-time "mask-then-open" double sharing. Because we skip the explicit
///     modulus switch (q = 2^k already has p | q, so Delta = q/p is exact),
///     the noise share <e>^Delta is masked and opened directly in (q, Delta),
///     not in any lifted ring.
#[derive(Clone, Copy, Debug)]
pub struct SpdzParams {
    pub k: u32,       // base modulus exponent  (PKE side)
    pub s: u32,       // statistical security bits (lift width)
    pub p: u128,      // plaintext modulus, power of two
    pub m_base: u128, // 2^k         — same as q below; kept for clarity
    pub m_share: u128,// 2^(k+s)     — secret-key sharing modulus (lifted)
    pub m_phi: u128,  // 2^(k+s)/p   — historical, currently unused by the protocol
    pub q: u128,      // 2^k         — PKE / ciphertext modulus
    pub delta: u128,  // q / p       — noise-share modulus for double sharing
}

impl SpdzParams {
    pub fn new(k: u32, s: u32, p: u128) -> Self {
        assert!(p.is_power_of_two(), "p must be a power of two");
        assert!(k + s <= 120, "k+s must fit comfortably in u128");
        assert!(k >= 1, "k must be at least 1");
        let m_share = 1u128 << (k + s);
        let m_base = 1u128 << k;
        let m_phi = m_share / p;
        let q = m_base;
        let delta = q / p;
        SpdzParams { k, s, p, m_base, m_share, m_phi, q, delta }
    }
}

// ---------------------------------------------------------------------------
// A scalar additive share
// ---------------------------------------------------------------------------

/// One party's additive share of a scalar over Z_{2^(k+s)}.
/// `x` is the party id (1..=n); `y` is the share value in [0, 2^(k+s)).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AddShare {
    pub x: u32,
    pub y: u128,
}

// ---------------------------------------------------------------------------
// Sampling helper
// ---------------------------------------------------------------------------

/// Uniform u128 in [0, modulus). Uses rand::Rng::gen_range, which handles the
/// modulo-bias correctly for arbitrary u128 bounds.
fn random_in_range<R: Rng>(modulus: u128, rng: &mut R) -> u128 {
    rng.gen_range(0..modulus)
}

// ---------------------------------------------------------------------------
// Share / reconstruct over an arbitrary 2^X modulus  (used for double sharing
// where `r` is sampled directly in the sharing modulus, no lift needed).
// ---------------------------------------------------------------------------

/// Additively share `value` (assumed already in [0, modulus)) across n parties.
/// First n-1 shares uniform in [0, modulus); last share makes the sum = value.
pub fn share(value: u128, n: usize, modulus: u128) -> Vec<AddShare> {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::from_entropy();
    share_with_rng(value, n, modulus, &mut rng)
}

/// As `share`, but with a caller-provided RNG (avoids re-seeding when sharing
/// many values in a row, e.g. all 256 coefficients of a polynomial).
pub fn share_with_rng<R: RngCore>(
    value: u128,
    n: usize,
    modulus: u128,
    rng: &mut R,
) -> Vec<AddShare> {
    assert!(n >= 2, "additive sharing needs at least 2 parties");
    debug_assert!(value < modulus, "value must be in [0, modulus)");

    // The RNG passed in is RngCore but rand::Rng::gen_range needs Rng; both
    // are blanket-implemented for everything that implements RngCore, but the
    // blanket Rng impl lives behind a trait import. Use that via a local Rng.
    use rand::Rng as _;

    let mut shares = Vec::with_capacity(n);
    let mut acc: u128 = 0;
    for i in 0..(n - 1) {
        let v = rng.gen_range(0..modulus);
        shares.push(AddShare { x: (i + 1) as u32, y: v });
        acc = (acc + v) % modulus;
    }
    // last share: (value - acc) mod modulus
    let last = if value >= acc { value - acc } else { modulus + value - acc };
    shares.push(AddShare { x: n as u32, y: last });

    debug_assert_eq!(
        shares.iter().fold(0u128, |a, s| (a + s.y) % modulus),
        value,
        "share-sum invariant"
    );
    shares
}

/// Sum additive shares mod `modulus`. Needs ALL n shares for the correct
/// value; fewer than n produces a uniform random result (the privacy property).
pub fn reconstruct(shares: &[AddShare], modulus: u128) -> u128 {
    shares.iter().fold(0u128, |acc, s| (acc + s.y) % modulus)
}

// ---------------------------------------------------------------------------
// Lifted sharing: secret in [0, 2^k) shared over Z_{2^(k+s)}
// ---------------------------------------------------------------------------

/// Share a secret in [0, 2^k) using SPDZ2k's statistically-hiding lift to
/// Z_{2^(k+s)}. After reconstruction (sum mod 2^(k+s)), reduce mod 2^k to get
/// the original secret.
pub fn share_lifted(secret_mod_2k: u128, n: usize, params: &SpdzParams) -> Vec<AddShare> {
    debug_assert!(
        secret_mod_2k < params.m_base,
        "secret must be in [0, 2^k); got {secret_mod_2k} >= {}",
        params.m_base
    );
    // Embed into the sharing ring as-is (value < 2^k < 2^(k+s)) and share.
    share(secret_mod_2k, n, params.m_share)
}

/// Reconstruct a lifted share back to the base modulus 2^k. Sums all shares
/// mod 2^(k+s), then reduces mod 2^k.
pub fn reconstruct_lifted(shares: &[AddShare], params: &SpdzParams) -> u128 {
    let lifted = reconstruct(shares, params.m_share);
    lifted % params.m_base
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_share_reconstruct_roundtrip_basic() {
        // Plain additive sharing over a power-of-two modulus, no lift.
        let modulus = 1u128 << 60;
        let v = 123_456_789_u128;
        let shares = share(v, 5, modulus);
        assert_eq!(shares.len(), 5);
        for (i, s) in shares.iter().enumerate() {
            assert_eq!(s.x, (i + 1) as u32);
            assert!(s.y < modulus);
        }
        assert_eq!(reconstruct(&shares, modulus), v);
    }

    #[test]
    fn test_lifted_share_recovers_base() {
        // Lifted sharing: secret in [0, 2^k), reconstruct mod 2^(k+s) then mod 2^k.
        let p = SpdzParams::new(20, 40, 2);
        let secret: u128 = 0xABCDE; // < 2^20
        let shares = share_lifted(secret, 4, &p);
        assert_eq!(shares.len(), 4);
        assert_eq!(reconstruct_lifted(&shares, &p), secret);

        // The shares themselves must be in [0, 2^(k+s)).
        for sh in &shares {
            assert!(sh.y < p.m_share);
        }
    }

    #[test]
    fn test_random_lifted_roundtrips() {
        // Many random secrets in [0, 2^k) round-trip through the lift.
        let p = SpdzParams::new(20, 40, 2);
        use rand::Rng;
        let mut rng = rand::thread_rng();
        for _ in 0..50 {
            let secret: u128 = rng.gen_range(0..p.m_base);
            let shares = share_lifted(secret, 3, &p);
            assert_eq!(reconstruct_lifted(&shares, &p), secret);
        }
    }

    #[test]
    fn test_partial_reveal_is_not_the_secret() {
        // Privacy sanity: n-1 shares should NOT equal the secret (overwhelming).
        // Not a formal hiding test, just a smoke check that the last share is
        // doing the work.
        let p = SpdzParams::new(20, 40, 2);
        let secret: u128 = 42;
        let shares = share_lifted(secret, 4, &p);
        let partial = reconstruct(&shares[..3], p.m_share);
        assert_ne!(partial % p.m_base, secret, "three of four shares shouldn't reveal the secret");
    }

    #[test]
    fn test_share_with_rng_reuse_is_deterministic() {
        // Same seeded RNG => same shares (regression / reproducibility).
        use rand::SeedableRng;
        let p = SpdzParams::new(20, 40, 2);
        let mut rng1 = rand::rngs::StdRng::seed_from_u64(42);
        let mut rng2 = rand::rngs::StdRng::seed_from_u64(42);
        let a = share_with_rng(7, 3, p.m_share, &mut rng1);
        let b = share_with_rng(7, 3, p.m_share, &mut rng2);
        assert_eq!(a, b);
    }
}