//! Block 3 — SPDZ2k Trusted Dealer (dishonest-majority additive sharing).
//!
//! Mirrors the Shamir `Dealer`, but for the 2^k PKE branch with the SPDZ2k
//! lift to 2^(k+s). Three offline responsibilities:
//!
//!   1. Run the 2^k PKE keygen (kpke::key_gen_2k).
//!   2. Lifted additive-share the secret key across n parties, into 2^(k+s)
//!      for statistical hiding.
//!   3. Generate `ell` independent additive double sharings (<d_j>^{q'},
//!      <d_j>^{mu'}) for masking. ell controls the threshold-failure
//!      amplification (Section 5.1 of the paper): majority decoding across
//!      ell candidates at receiver_reconstruct time.
//!
//! Notation matches the paper:
//!   q       : original PKE modulus (= 2^k = m_base)
//!   q'      : modulus after switching (here q' = q since q is a power of 2)
//!   mu      : q / p  (was 'delta' in earlier versions)
//!   mu'     : q' / p (= mu here)
//!
//! Differences from the Shamir dealer:
//!   * NO threshold t — dishonest majority means t = n-1; reconstruction
//!     needs ALL n parties.
//!   * Sharing is ADDITIVE, lifted from 2^k to 2^(k+s).
//!   * Double sharing is over POWERS OF TWO (q' and mu'), both additive.
//!   * Secret key is in COEFFICIENT form already (no NTT in the 2^k branch).

use crate::additive_2k::{self, AddShare, SpdzParams};
use crate::additive_ring::{self, VectorShare128};
use crate::kpke;
use crate::params::*;

// ---------------------------------------------------------------------------
// Output types
// ---------------------------------------------------------------------------

pub type EncryptionKey<const K: usize> = kpke::KpkeEncryptionKey<K>;

/// One party's additive double-share of the ell masking polynomials
/// d_1, ..., d_ell, over BOTH base rings (q' and mu'). Coefficient form,
/// 256 coefficients per ring per parallel sharing.
///
/// `q_prime[j]` and `mu_prime[j]` are this party's share of d_j over
/// Z_{q'} and Z_{mu'} respectively. Their (coefficient-wise) sums over all
/// n parties equal d_j mod q' and mod mu'.
#[derive(Clone, Debug)]
pub struct DoubleShare {
    pub party_id: u32,
    pub q_prime: Vec<[u128; 256]>,  // length ell; shares over Z_{q'}
    pub mu_prime: Vec<[u128; 256]>, // length ell; shares over Z_{mu'}
}

impl DoubleShare {
    /// Number of parallel double sharings this struct carries.
    #[inline]
    pub fn ell(&self) -> usize {
        debug_assert_eq!(self.q_prime.len(), self.mu_prime.len());
        self.q_prime.len()
    }
}

/// Output of key generation + lifted secret-key sharing.
pub struct KeyShares<const K: usize> {
    pub ek: EncryptionKey<K>,
    pub sk_shares: Vec<VectorShare128<K>>,
}

// ---------------------------------------------------------------------------
// Dealer
// ---------------------------------------------------------------------------

/// SPDZ2k trusted dealer.
pub struct DealerSpdz {
    pub n: usize,
    pub thr: SpdzParams,
}

impl DealerSpdz {
    /// Create a dealer for n parties at the given (k, s, p). Dishonest-majority,
    /// so the implicit threshold is t = n-1 (reconstruction needs all n).
    pub fn new(n: usize, k: u32, s: u32, p: u128) -> Self {
        assert!(n >= 2, "need at least 2 parties");
        DealerSpdz { n, thr: SpdzParams::new(k, s, p) }
    }

    /// Accessor mirroring the Shamir dealer for symmetric code paths.
    pub fn params(&self) -> &SpdzParams { &self.thr }

    /// Run the 2^k PKE keygen and lifted-additive-share the secret key.
    pub fn generate_keypair<PARAMS: MlKemParams>(&self) -> KeyShares<{ PARAMS::K }>
    where
        [(); 960 * PARAMS::K + 32]:,
        [(); 1920 * PARAMS::K + 96]:,
        [(); PARAMS::K]:,
        [(); PARAMS::ETA_2]:,
        [(); 64 * PARAMS::ETA_1]:,
        [(); 32 * (PARAMS::D_U * PARAMS::K + PARAMS::D_V)]:,
    {
        let (ek, s) = kpke::key_gen_2k::<PARAMS>();
        let sk_shares =
            additive_ring::share_vector_lifted::<{ PARAMS::K }>(&s, self.n, &self.thr);
        KeyShares { ek, sk_shares }
    }

    /// Generate `ell` independent additive double sharings.
    ///
    /// Each d_j is sampled with coefficients uniform in [0, mu') and additively
    /// shared independently over Z_{q'} and Z_{mu'}. The ell sharings are
    /// packed per-party: party i's returned `DoubleShare` holds Vec-of-arrays
    /// for q_prime and mu_prime, both of length ell.
    pub fn generate_double_sharing(&self, ell: usize) -> Vec<DoubleShare> {
        assert!(ell >= 1, "ell must be >= 1");
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::from_entropy();

        // Initialize n empty per-party DoubleShares with capacity ell.
        let mut out: Vec<DoubleShare> = (0..self.n)
            .map(|i| DoubleShare {
                party_id: (i + 1) as u32,
                q_prime: Vec::with_capacity(ell),
                mu_prime: Vec::with_capacity(ell),
            })
            .collect();

        for _ in 0..ell {
            // d_j: 256 coefficients uniform in [0, mu').
            let mut d_coeffs = [0u128; 256];
            for c in 0..256 {
                d_coeffs[c] = rng.gen_range(0..self.thr.mu_prime);
            }

            // Additive shares of d_j over q' and over mu'.
            let q_prime_shares_per_coeff: Vec<Vec<AddShare>> = (0..256)
                .map(|c| {
                    additive_2k::share_with_rng(d_coeffs[c], self.n, self.thr.q_prime, &mut rng)
                })
                .collect();
            let mu_prime_shares_per_coeff: Vec<Vec<AddShare>> = (0..256)
                .map(|c| {
                    additive_2k::share_with_rng(d_coeffs[c], self.n, self.thr.mu_prime, &mut rng)
                })
                .collect();

            // Regroup by party: party i gets one [u128; 256] for q' and one for mu'.
            for i in 0..self.n {
                let mut q_p = [0u128; 256];
                let mut m_p = [0u128; 256];
                for c in 0..256 {
                    q_p[c] = q_prime_shares_per_coeff[c][i].y;
                    m_p[c] = mu_prime_shares_per_coeff[c][i].y;
                }
                out[i].q_prime.push(q_p);
                out[i].mu_prime.push(m_p);
            }
        }

        out
    }
}
