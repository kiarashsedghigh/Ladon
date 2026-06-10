//! Block 2 — Shamir secret sharing for the project's Ring and Vector<K> types.
//!
//! This lifts the scalar F_q sharing in `shamir.rs` to whole polynomials and
//! polynomial vectors by sharing each coefficient INDEPENDENTLY. The secret
//! key produced by `kpke::key_gen` is a `Vector<K>` (K rings of 256 u32
//! coefficients each), so `share_vector` is the function the dealer (Block 3)
//! will call.
//!
//! Representation note
//! -------------------
//! Shamir is applied in COEFFICIENT form (RingRepresentation::Degree255), NOT
//! NTT form, matching the Python reference (`from_ntt()` before sharing) and
//! Ajax Fig. 10. The dealer is responsible for calling `inverse_ntt` on the
//! secret key before handing it here. All share Rings are tagged Degree255.
//!
//! Transpose intuition
//! -------------------
//! Scalar `share(coeff, t, n)` returns n shares of ONE coefficient. A Ring has
//! 256 coefficients, so sharing a Ring means running `share` 256 times and
//! regrouping ("transposing") the results so that party j receives one Ring
//! holding the j-th share of every coefficient.

use crate::ring::{Ring, RingRepresentation, Vector};
use crate::shamir;

/// One party's Shamir share of a Ring: the share polynomial (one coefficient
/// per slot) together with the evaluation point x = party id.
#[derive(Clone, Debug, PartialEq)]
pub struct RingShare {
    pub x: u32,
    pub ring: Ring,
}

/// One party's Shamir share of a Vector<K>: K ring-shares, all at the same
/// evaluation point x.
#[derive(Clone, Debug)]
pub struct VectorShare<const K: usize> {
    pub x: u32,
    pub vector: Vector<K>,
}

/// Shamir-share a single Ring across n parties with threshold t.
///
/// Returns n RingShares (party ids 1..=n). Each coefficient is shared with an
/// independent random degree-t polynomial; party j's ring collects the j-th
/// share of all 256 coefficients. Any t+1 ring-shares reconstruct the Ring.
///
/// The share rings inherit the SAME representation tag as `secret` (NTT or
/// Degree255). Sharing is F_q-linear and commutes with the NTT, so sharing a
/// secret that is already in NTT form yields NTT-form shares on which parties
/// can run `Ring::mult` directly during decryption.
pub fn share_ring(secret: &Ring, t: usize, n: usize) -> Vec<RingShare> {
    let zero = match secret.t {
        RingRepresentation::NTT => Ring::ZEROES_NTT,
        RingRepresentation::Degree255 => Ring::ZEROES_DEGREE255,
    };

    // Start with n zero rings (one per party), tagged to match the secret.
    let mut out: Vec<RingShare> = (1..=n as u32)
        .map(|x| RingShare { x, ring: zero.clone() })
        .collect();

    // Share each coefficient independently and scatter into the per-party rings.
    // Seed ONE RNG for the whole ring (256 coefficients) instead of reseeding
    // from entropy per coefficient — that reseeding was the test slowness.
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::from_entropy();
    for coeff_idx in 0..256 {
        let shares = shamir::share_with_rng(secret.data[coeff_idx], t, n, &mut rng);
        // shares[p] corresponds to party id (p+1), same order as `out`.
        for (party_idx, sh) in shares.iter().enumerate() {
            debug_assert_eq!(sh.x, out[party_idx].x); // points line up
            out[party_idx].ring.data[coeff_idx] = sh.y;
        }
    }

    out
}

/// Reconstruct a Ring from t+1 (or more) RingShares via coefficient-wise
/// Lagrange interpolation at zero. The result inherits the shares' domain tag.
pub fn reconstruct_ring(shares: &[RingShare]) -> Ring {
    let mut result = match shares[0].ring.t {
        RingRepresentation::NTT => Ring::ZEROES_NTT,
        RingRepresentation::Degree255 => Ring::ZEROES_DEGREE255,
    };

    for coeff_idx in 0..256 {
        // Gather this coefficient's scalar shares across the given parties.
        let scalar_shares: Vec<shamir::Share> = shares
            .iter()
            .map(|rs| shamir::Share { x: rs.x, y: rs.ring.data[coeff_idx] })
            .collect();
        result.data[coeff_idx] = shamir::reconstruct(&scalar_shares);
    }

    result
}

/// Shamir-share a Vector<K> (the secret key) across n parties, threshold t.
///
/// Returns n VectorShares (party ids 1..=n). Each of the K rings is shared
/// independently via `share_ring`; the resulting per-party ring-shares are
/// grouped into one Vector<K> per party (all sharing the same eval point x).
pub fn share_vector<const K: usize>(
    secret: &Vector<K>,
    t: usize,
    n: usize,
) -> Vec<VectorShare<K>> {
    // Share each of the K rings; ring_shares[k] is a Vec<RingShare> of len n.
    let per_ring: Vec<Vec<RingShare>> =
        (0..K).map(|k| share_ring(&secret.data[k], t, n)).collect();

    // Regroup by party: party j gets ring k's j-th share, for all k.
    // Each ring already carries the correct domain tag from share_ring, so we
    // build the container and overwrite every slot.
    (0..n)
        .map(|party_idx| {
            let x = per_ring[0][party_idx].x;
            let mut v = Vector::<K>::new_degree255(); // tags overwritten below
            for k in 0..K {
                debug_assert_eq!(per_ring[k][party_idx].x, x); // consistent point
                v.data[k] = per_ring[k][party_idx].ring.clone();
            }
            VectorShare { x, vector: v }
        })
        .collect()
}

/// Reconstruct a Vector<K> from t+1 (or more) VectorShares.
pub fn reconstruct_vector<const K: usize>(shares: &[VectorShare<K>]) -> Vector<K> {
    let mut result = Vector::<K>::new_degree255();

    for k in 0..K {
        // Pull out ring k's shares across the given parties.
        let ring_shares: Vec<RingShare> = shares
            .iter()
            .map(|vs| RingShare { x: vs.x, ring: vs.vector.data[k].clone() })
            .collect();
        result.data[k] = reconstruct_ring(&ring_shares);
    }

    result
}
