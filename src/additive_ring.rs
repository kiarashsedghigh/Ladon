//! Block 2 — Additive Ring/Vector sharing over Z_{2^(k+s)} for SPDZ2k.
//!
//! Lifts the scalar layer (additive_2k) to whole polynomials and polynomial
//! vectors by sharing each coefficient INDEPENDENTLY. The secret key produced
//! by the 2^k PKE keygen is a Vector<K> with u32 coefficients in [0, 2^k);
//! shares of it live in Z_{2^(k+s)}, which doesn't fit u32 — so per-coefficient
//! share storage is u128.
//!
//! Storage types
//! -------------
//! Input  (secret) : Ring / Vector<K> with u32 coefficients in [0, 2^k).
//! Output (shares) : RingShare128 / VectorShare128 with u128 coefficients in
//!                   [0, 2^(k+s)).
//!
//! Transpose intuition (same as shamir_ring): scalar `share` returns n shares
//! of ONE coefficient. A Ring has 256 coefficients, so we run share 256 times
//! and regroup so party j gets one RingShare128 containing the j-th share of
//! every coefficient.

use crate::additive_2k::{self, AddShare, SpdzParams};
use crate::ring::{Ring, Vector};

// ---------------------------------------------------------------------------
// Share containers (u128 coefficients, modulus 2^(k+s))
// ---------------------------------------------------------------------------

/// One party's additive share of a Ring: 256 u128 coefficients, all in
/// [0, 2^(k+s)), together with the party id. Inhabitants of this struct are
/// always in the SHARING ring (modulus 2^(k+s)), not the base ring 2^k.
#[derive(Clone, Debug, PartialEq)]
pub struct RingShare128 {
    pub x: u32,
    pub data: [u128; 256],
}

impl RingShare128 {
    pub fn zero(x: u32) -> Self {
        RingShare128 { x, data: [0u128; 256] }
    }
}

/// One party's additive share of a Vector<K>: K ring-shares, all at the same
/// party id x.
#[derive(Clone, Debug)]
pub struct VectorShare128<const K: usize> {
    pub x: u32,
    pub rings: [RingShare128; K],
}

// ---------------------------------------------------------------------------
// Ring-level: share / reconstruct
// ---------------------------------------------------------------------------

/// Additively share a Ring (coefficient form, u32 values in [0, 2^k)) across
/// n parties, lifted into Z_{2^(k+s)}. Returns n RingShare128s (ids 1..=n).
/// Sum of all n shares mod 2^(k+s), reduced mod 2^k, recovers the original.
pub fn share_ring_lifted(secret: &Ring, n: usize, params: &SpdzParams) -> Vec<RingShare128> {
    // One RNG seeded ONCE for the whole ring (256 coefficients * (n-1) draws).
    // Reseeding from entropy per coefficient was the slow-test problem in the
    // shamir branch; same pattern here.
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::from_entropy();

    let mut out: Vec<RingShare128> = (1..=n as u32).map(RingShare128::zero).collect();

    for coeff_idx in 0..256 {
        let secret_coeff = secret.data[coeff_idx] as u128;
        debug_assert!(secret_coeff < params.m_base);
        let scalar_shares: Vec<AddShare> =
            additive_2k::share_with_rng(secret_coeff, n, params.m_share, &mut rng);
        // shares[p] corresponds to party id (p+1), same order as `out`.
        for (party_idx, sh) in scalar_shares.iter().enumerate() {
            debug_assert_eq!(sh.x, out[party_idx].x);
            out[party_idx].data[coeff_idx] = sh.y;
        }
    }
    out
}

/// Reconstruct a Ring (mod 2^k, coefficient form) from a full set of n shares.
/// Sums coefficient-wise mod 2^(k+s), then reduces each coefficient mod 2^k.
/// Result fits u32 since 2^k <= 2^32 in our regime.
pub fn reconstruct_ring_lifted(shares: &[RingShare128], params: &SpdzParams) -> Ring {
    let mut out = Ring::ZEROES_DEGREE255;
    for coeff_idx in 0..256 {
        let sum = shares
            .iter()
            .fold(0u128, |acc, rs| (acc + rs.data[coeff_idx]) % params.m_share);
        let base = sum % params.m_base;
        debug_assert!(base <= u32::MAX as u128);
        out.data[coeff_idx] = base as u32;
    }
    out
}

/// Reconstruct a Ring over an ARBITRARY power-of-two modulus, summing the
/// per-coefficient shares mod the given modulus. Used by double sharing where
/// the modulus may be 2^(k+s) (full) or 2^(k+s)/p (the Phi sub-ring). The
/// returned Ring stores the (possibly wide) values truncated to u32 — caller
/// must know what range to expect.
pub fn reconstruct_ring_mod(shares: &[RingShare128], modulus: u128) -> [u128; 256] {
    let mut out = [0u128; 256];
    for coeff_idx in 0..256 {
        out[coeff_idx] = shares
            .iter()
            .fold(0u128, |acc, rs| (acc + rs.data[coeff_idx]) % modulus);
    }
    out
}

// ---------------------------------------------------------------------------
// Vector<K> level: share / reconstruct
// ---------------------------------------------------------------------------

/// Additively share a Vector<K> (the secret key, K rings of u32 coefficients
/// in [0, 2^k)) across n parties, lifted to Z_{2^(k+s)}. Each of the K rings
/// is shared independently via `share_ring_lifted`, then regrouped so each
/// party holds one VectorShare128<K>.
pub fn share_vector_lifted<const K: usize>(
    secret: &Vector<K>,
    n: usize,
    params: &SpdzParams,
) -> Vec<VectorShare128<K>> {
    // Share each of the K rings; per_ring[k] is a Vec<RingShare128> of len n.
    let per_ring: Vec<Vec<RingShare128>> =
        (0..K).map(|k| share_ring_lifted(&secret.data[k], n, params)).collect();

    // Regroup by party: party j gets ring k's j-th share, for all k.
    (0..n)
        .map(|party_idx| {
            let x = per_ring[0][party_idx].x;
            // Build the [RingShare128; K] from the per-ring grouping.
            let rings: [RingShare128; K] = core::array::from_fn(|k| {
                debug_assert_eq!(per_ring[k][party_idx].x, x);
                per_ring[k][party_idx].clone()
            });
            VectorShare128 { x, rings }
        })
        .collect()
}

/// Reconstruct a Vector<K> from a full set of n VectorShare128s.
pub fn reconstruct_vector_lifted<const K: usize>(
    shares: &[VectorShare128<K>],
    params: &SpdzParams,
) -> Vector<K> {
    let mut result = Vector::<K>::new_degree255();
    for k in 0..K {
        // Pull out ring k's shares across the n parties.
        let ring_shares: Vec<RingShare128> = shares.iter().map(|vs| vs.rings[k].clone()).collect();
        result.data[k] = reconstruct_ring_lifted(&ring_shares, params);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ring(seed: u32, k_bits: u32) -> Ring {
        // Deterministic non-trivial ring with coeffs in [0, 2^k_bits).
        let mask: u32 = (1u32 << k_bits) - 1;
        let mut r = Ring::ZEROES_DEGREE255;
        for i in 0..256 {
            let v = seed
                .wrapping_mul(2_654_435_761)
                .wrapping_add((i as u32).wrapping_mul(40_503));
            r.data[i] = v & mask;
        }
        r
    }

    #[test]
    fn test_share_reconstruct_ring_lifted() {
        let params = SpdzParams::new(20, 40, 2);
        let secret = make_ring(7, params.k);
        let n = 4;
        let shares = share_ring_lifted(&secret, n, &params);
        assert_eq!(shares.len(), n);
        // Eval points 1..=n
        for (i, s) in shares.iter().enumerate() {
            assert_eq!(s.x, (i + 1) as u32);
            // Every share coefficient sits in the sharing ring.
            for c in 0..256 {
                assert!(s.data[c] < params.m_share);
            }
        }
        let rec = reconstruct_ring_lifted(&shares, &params);
        assert_eq!(rec, secret);
    }

    #[test]
    fn test_share_reconstruct_vector_lifted() {
        const K: usize = 3;
        let params = SpdzParams::new(20, 40, 2);
        let mut secret = Vector::<K>::new_degree255();
        for k in 0..K {
            secret.data[k] = make_ring(100 + k as u32, params.k);
        }

        let n = 5;
        let shares = share_vector_lifted::<K>(&secret, n, &params);
        assert_eq!(shares.len(), n);
        for (i, s) in shares.iter().enumerate() {
            assert_eq!(s.x, (i + 1) as u32);
        }

        let rec = reconstruct_vector_lifted::<K>(&shares, &params);
        assert_eq!(rec, secret);
    }

    #[test]
    fn test_partial_reveal_does_not_match_secret() {
        // n-1 shares should overwhelmingly NOT equal the secret (sanity smoke).
        let params = SpdzParams::new(20, 40, 2);
        let secret = make_ring(13, params.k);
        let n = 4;
        let shares = share_ring_lifted(&secret, n, &params);

        let partial: [u128; 256] =
            super::reconstruct_ring_mod(&shares[..n - 1], params.m_share);
        // Compare coefficient-wise reduced to 2^k against the secret.
        let mut equal = true;
        for c in 0..256 {
            if (partial[c] % params.m_base) as u32 != secret.data[c] {
                equal = false;
                break;
            }
        }
        assert!(!equal, "n-1 shares should not reproduce the secret");
    }
}