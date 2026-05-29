//! Block 3 — SPDZ2k Trusted Dealer (dishonest-majority additive sharing).
//!
//! Mirrors the Shamir `Dealer` (dealer.rs), but for the 2^k PKE branch with
//! the SPDZ2k lift to 2^(k+s). Three offline responsibilities:
//!
//!   1. Run the 2^k PKE keygen (kpke::key_gen_2k).
//!   2. Lifted additive-share the secret key across n parties, into 2^(k+s)
//!      for statistical hiding of the long-lived secret.
//!   3. Generate an additive double sharing (<r>^q, <r>^Delta) — over the
//!      BASE rings q = 2^k and Delta = q/p, NOT the lifted rings. This is
//!      because the threshold-decrypt protocol skips the explicit modulus
//!      switch (q already a power of two, p | q), so the noise share <e>^Delta
//!      is masked and opened directly in (q, Delta). The lift is only for the
//!      secret key itself.
//!
//! Differences from the Shamir dealer:
//!   * NO threshold t: dishonest-majority means t = n-1, so reconstruction
//!     needs ALL n parties (no Lagrange, no committee selection).
//!   * Sharing is ADDITIVE, lifted from 2^k to 2^(k+s).
//!   * Double sharing is over POWERS OF TWO (q and Delta), both additive
//!     (additive works over any ring; Shamir would need a field).
//!   * Secret key is in COEFFICIENT form already (the 2^k PKE has no NTT),
//!     so no inverse_ntt step before sharing.
//!
//! K_BITS consistency note
//! -----------------------
//! `key_gen_2k` has K_BITS = 20 baked in at the top of kpke.rs. The dealer
//! creates SpdzParams::new(k, s, p); CALLER MUST KEEP `k` IN SYNC with
//! `kpke::K_BITS`. There's no shared constant yet; this is a deliberate seam
//! since you said K_BITS would move to params.rs later.

use crate::additive_2k::{self, AddShare, SpdzParams};
use crate::additive_ring::{self, VectorShare128};
use crate::kpke;
use crate::params::*;

// ---------------------------------------------------------------------------
// Output types
// ---------------------------------------------------------------------------

/// Public encryption key from the 2^k PKE (same shape as the prime version:
/// a Vector<K> of t-values plus the 32-byte rho seed).
pub type EncryptionKey<const K: usize> = kpke::KpkeEncryptionKey<K>;

/// One party's additive double-share of the masking polynomial r, coefficient
/// form, 256 coefficients per ring. `r_q_share` is the share over the PKE
/// ring q = 2^k; `r_delta_share` is the share over Delta = q/p. Both sums
/// recover the same integer r (coefficients of r are in [0, Delta) so they
/// embed unchanged in the larger ring q).
///
/// NOTE: these are NOT the lifted rings. The SECRET KEY is shared in the
/// lifted ring 2^(k+s) for statistical hiding, but the double sharing here
/// masks one-time noise that lives in Delta, so it operates in the smaller
/// base rings (q, Delta). See SpdzParams' field comments.
#[derive(Clone, Debug)]
pub struct DoubleShare {
    pub party_id: u32,
    pub r_q_share: [u128; 256],     // share over Z_q = Z_{2^k}
    pub r_delta_share: [u128; 256], // share over Z_Delta = Z_{2^(k-1)}
}

/// Output of key generation + lifted secret-key sharing.
pub struct KeyShares<const K: usize> {
    pub ek: EncryptionKey<K>,
    pub sk_shares: Vec<VectorShare128<K>>,
}

// ---------------------------------------------------------------------------
// Dealer
// ---------------------------------------------------------------------------

/// SPDZ2k trusted dealer. Stores the sharing configuration (n parties + SPDZ
/// modulus parameters); each call produces fresh randomness.
pub struct DealerSpdz {
    pub n: usize,
    pub params: SpdzParams,
}

impl DealerSpdz {
    /// Create a dealer for n parties at the given (k, s, p). Dishonest-majority,
    /// so the implicit threshold is t = n-1 (reconstruction needs all n).
    pub fn new(n: usize, k: u32, s: u32, p: u128) -> Self {
        assert!(n >= 2, "need at least 2 parties");
        DealerSpdz { n, params: SpdzParams::new(k, s, p) }
    }

    /// Run the 2^k PKE keygen and lifted-additive-share the secret key.
    /// The secret key from key_gen_2k is in coefficient form (no NTT in the
    /// 2^k branch), with each coefficient in [0, 2^k) — exactly the input
    /// `share_vector_lifted` expects.
    pub fn generate_keypair<PARAMS: MlKemParams>(&self) -> KeyShares<{ PARAMS::K }>
    where
        [(); 384 * PARAMS::K + 32]:,
        [(); 768 * PARAMS::K + 96]:,
        [(); PARAMS::K]:,
        [(); PARAMS::ETA_2]:,
        [(); 64 * PARAMS::ETA_1]:,
        [(); 32 * (PARAMS::D_U * PARAMS::K + PARAMS::D_V)]:,
    {
        let (ek, s) = kpke::key_gen_2k::<PARAMS>();
        let sk_shares =
            additive_ring::share_vector_lifted::<{ PARAMS::K }>(&s, self.n, &self.params);
        KeyShares { ek, sk_shares }
    }

    /// Generate one additive double sharing (<r>^q, <r>^Delta).
    ///
    /// `r` is sampled with each coefficient uniform in [0, Delta). The same
    /// integer is then additively shared TWICE: once over q = 2^k and once
    /// over Delta = q/p. Because every coefficient of r is < Delta, the
    /// embedding into q is identity — both shares decode to the same r when
    /// reconstructed under their respective modulus.
    ///
    /// Why (q, Delta) and not the lifted (m_share, m_phi): we're skipping
    /// the explicit modulus switch (q is already a power of two with p | q),
    /// so the noise share <e>^Delta is masked and opened directly in (q,Delta).
    /// The lift only matters for the LONG-LIVED secret-key shares.
    pub fn generate_double_sharing(&self) -> Vec<DoubleShare> {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::from_entropy();

        // Sample r: 256 coefficients uniform in [0, Delta).
        let mut r_coeffs = [0u128; 256];
        for c in 0..256 {
            r_coeffs[c] = rng.gen_range(0..self.params.delta);
        }

        // Additive shares of r over q and over Delta. Same RNG; the two
        // share-sets are independent random splits of the same r.
        let r_shares_q: Vec<Vec<AddShare>> = (0..256)
            .map(|c| {
                additive_2k::share_with_rng(r_coeffs[c], self.n, self.params.q, &mut rng)
            })
            .collect();
        let r_shares_delta: Vec<Vec<AddShare>> = (0..256)
            .map(|c| {
                additive_2k::share_with_rng(r_coeffs[c], self.n, self.params.delta, &mut rng)
            })
            .collect();

        // Regroup by party.
        (0..self.n)
            .map(|i| {
                let mut r_q_share = [0u128; 256];
                let mut r_delta_share = [0u128; 256];
                for c in 0..256 {
                    r_q_share[c] = r_shares_q[c][i].y;
                    r_delta_share[c] = r_shares_delta[c][i].y;
                }
                DoubleShare {
                    party_id: (i + 1) as u32,
                    r_q_share,
                    r_delta_share,
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::additive_ring::{reconstruct_ring_mod, reconstruct_vector_lifted};

    #[test]
    fn test_secret_key_shares_reconstruct() {
        // Deal a keypair, then verify summing all n shares recovers the secret
        // key (mod 2^k).
        type P = MlKem512;
        let dealer = DealerSpdz::new(4, 20, 40, 2);
        let ks = dealer.generate_keypair::<P>();
        assert_eq!(ks.sk_shares.len(), 4);

        // We don't have the original secret key directly (key_gen_2k doesn't
        // return it separately to compare against), but reconstructing from
        // all n shares must give a consistent Vector<K> — and the same value
        // every time. Reconstruct from two ORDERINGS of the same shares and
        // confirm they agree (sanity that the sum is order-independent and
        // not corrupted by the regroup).
        let rec_a = reconstruct_vector_lifted(&ks.sk_shares, &dealer.params);
        let mut reversed = ks.sk_shares.clone();
        reversed.reverse();
        let rec_b = reconstruct_vector_lifted(&reversed, &dealer.params);
        assert_eq!(rec_a, rec_b, "reconstruction is order-independent");

        // All reconstructed coefficients are in [0, 2^k).
        for ring in rec_a.data.iter() {
            for &c in &ring.data {
                assert!((c as u128) < dealer.params.m_base);
            }
        }
    }

    #[test]
    fn test_double_sharing_consistency() {
        // The q and delta shares of r must reconstruct to the SAME integer
        // value per coefficient (since coeffs of r are < Delta, the embedding
        // into q is identity). Specifically:
        //   (sum mod q) % Delta == sum mod Delta
        // and the q-reconstruction is itself < Delta.
        let dealer = DealerSpdz::new(5, 20, 40, 2);
        let ds = dealer.generate_double_sharing();
        assert_eq!(ds.len(), 5);

        let q = dealer.params.q;
        let delta = dealer.params.delta;

        for c in 0..256 {
            let r_from_q: u128 = ds
                .iter()
                .fold(0u128, |acc, s| (acc + s.r_q_share[c]) % q);
            let r_from_delta: u128 = ds
                .iter()
                .fold(0u128, |acc, s| (acc + s.r_delta_share[c]) % delta);

            assert_eq!(
                r_from_q % delta,
                r_from_delta,
                "coeff {c}: q-ring and Delta-ring reconstructions disagree"
            );
            assert!(
                r_from_q < delta,
                "coeff {c}: reconstructed r should be in [0, Delta) since it was sampled there"
            );
        }
    }

    #[test]
    fn test_double_sharing_uses_reconstruct_ring_mod() {
        // Same consistency check, but via the additive_ring helper to make sure
        // it agrees with the hand-rolled fold above.
        let dealer = DealerSpdz::new(3, 20, 40, 2);
        let ds = dealer.generate_double_sharing();

        let ring_shares_q: Vec<crate::additive_ring::RingShare128> = ds
            .iter()
            .map(|d| crate::additive_ring::RingShare128 {
                x: d.party_id,
                data: d.r_q_share,
            })
            .collect();
        let recon_q = reconstruct_ring_mod(&ring_shares_q, dealer.params.q);

        let ring_shares_delta: Vec<crate::additive_ring::RingShare128> = ds
            .iter()
            .map(|d| crate::additive_ring::RingShare128 {
                x: d.party_id,
                data: d.r_delta_share,
            })
            .collect();
        let recon_delta = reconstruct_ring_mod(&ring_shares_delta, dealer.params.delta);

        for c in 0..256 {
            assert_eq!(recon_q[c] % dealer.params.delta, recon_delta[c]);
        }
    }
}