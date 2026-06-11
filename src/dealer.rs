//! Block 3 — Trusted Dealer (struct form).
//!
//! Combines the offline responsibilities of the trusted dealer:
//!   1. Run K-PKE key generation.
//!   2. Shamir-share the (NTT-form) secret key across n parties (degree t-1).
//!   3. Generate additive double sharings (<r>^{q'}, <r>^{mu'}) across the
//!      ACTIVE COMMITTEE of size t (not n).
//!
//! Two share-class sizes:
//!   * Shamir key shares: n total, any t reconstruct (any t-1 reveal nothing).
//!     Used for offline storage robustness — even if up to t-1 long-term
//!     shareholders are corrupted, the secret key remains hidden.
//!   * Additive double sharings: t total, one per active-committee slot.
//!     Additive sharing is intrinsically all-of-all: the t shares must be
//!     summed in full to reconstruct the masking polynomial r. The active
//!     committee is fixed at sharing time and corresponds to the FIRST t
//!     party ids (1..=t).
//!
//! Notation (matching the paper):
//!   q       : original ciphertext modulus (= params::Q)
//!   q'      : new modulus after switching; chosen as 2^round(log2(q))
//!             so that q' < q and the protocol mod-switches q -> q'.
//!   mu      : q / p   — half-modulus in the original ring (for p = 2)
//!   mu'     : q' / p  — half-modulus in the switched ring (for p = 2)

use crate::kpke;
use crate::params::*;
use crate::ring::Vector;
use crate::shamir_poly_ring::{self, VectorShare};

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

// ---------------------------------------------------------------------------
// Threshold-decryption parameters (mu, q', mu') derived from q and p.
// ---------------------------------------------------------------------------

/// Parameters for the mask-then-open threshold decryption, derived from the
/// ciphertext modulus q and plaintext modulus p.
#[derive(Clone, Copy, Debug)]
pub struct ThrParams {
    pub q: u64,
    pub p: u64,
    pub mu: u64,        // q / p
    pub q_prime: u64,   // 2^round(log2(q)); paper's q'
    pub mu_prime: u64,  // q' / p; paper's mu'
}

impl ThrParams {
    /// Build from q (= params::Q) and a power-of-two plaintext modulus p.
    /// q' is chosen as the power of 2 nearest to q (typically just below).
    /// Requires p | q'.
    pub fn new(p: u64) -> Self {
        assert!(p.is_power_of_two(), "p must be a power of two");
        let q = Q64;

        // q' = 2^round(log2(q)). For q = 687659009 this is 2^29 = 536870912.
        let log2_q = (q as f64).log2();
        let k = log2_q.round() as u32;
        let q_prime: u64 = 1u64 << k;
        assert!(q_prime <= q, "q' must be <= q for modulus switching down");
        assert!(q_prime % p == 0, "p must divide q'");

        ThrParams {
            q,
            p,
            mu: q / p,
            q_prime,
            mu_prime: q_prime / p,
        }
    }
}

// ---------------------------------------------------------------------------
// Additive double sharing over two rings (Z_{q'}, Z_{mu'}).
// ---------------------------------------------------------------------------

/// One active-committee member's additive share of the masking polynomials
/// r_1, ..., r_ell, over BOTH rings. Coefficient form, 256 coefficients each.
///
/// For majority decoding (Section 5.1 of the paper), the dealer produces `ell`
/// parallel double sharings; the `q_prime`/`mu_prime` Vecs each have length
/// `ell`, where `q_prime[j]` and `mu_prime[j]` are this member's share of r_j
/// over Z_{q'} and Z_{mu'} respectively. Their (coefficient-wise) sums over
/// the t active members equal r_j mod q' and mod mu' respectively.
#[derive(Clone, Debug)]
pub struct DoubleShare {
    pub party_id: u32,
    pub q_prime: Vec<[u64; 256]>,  // length ell; shares over Z_{q'}
    pub mu_prime: Vec<[u64; 256]>, // length ell; shares over Z_{mu'}
}

impl DoubleShare {
    #[inline]
    pub fn ell(&self) -> usize {
        debug_assert_eq!(self.q_prime.len(), self.mu_prime.len());
        self.q_prime.len()
    }
}

/// Additively share a coefficient vector `secret` (length 256) over modulus m
/// across `t_active` parties, using the provided RNG. Returns t_active arrays
/// whose coefficient-wise sum mod m equals `secret`.
fn additive_share_coeffs<R: Rng>(
    secret: &[u64; 256],
    t_active: usize,
    m: u64,
    rng: &mut R,
) -> Vec<[u64; 256]> {
    assert!(t_active >= 2, "additive sharing needs >= 2 parties");
    let mut shares = vec![[0u64; 256]; t_active];

    for c in 0..256 {
        let mut acc = 0u64;
        // First t_active - 1 shares uniform in [0, m); last makes sum = secret.
        for share in shares.iter_mut().take(t_active - 1) {
            let r = rng.gen_range(0..m);
            share[c] = r;
            acc = (acc + r) % m;
        }
        let last = (secret[c] % m + m - acc) % m;
        shares[t_active - 1][c] = last;
    }
    shares
}

// ---------------------------------------------------------------------------
// Dealer
// ---------------------------------------------------------------------------

pub type EncryptionKey<const K: usize> = kpke::KpkeEncryptionKey<K>;

/// Trusted dealer holding the threshold configuration.
///
/// Field semantics:
///   * `t` : min-to-decrypt. The active committee has exactly `t` members
///           (the first `t` party ids, 1..=t). Additive double sharings
///           are produced across these `t` members. The Shamir polynomial
///           has degree `t - 1`, so any `t` Shamir shares reconstruct sk.
///   * `n` : total parties to whom Shamir key shares are dealt. Provides
///           offline robustness — any `t-1` long-term Shamir shareholders
///           cannot learn sk. The online committee is still the first `t`.
pub struct Dealer {
    pub t: usize,
    pub n: usize,
    pub thr: ThrParams,
}

/// Output of key generation + secret-key sharing.
pub struct KeyShares<const K: usize> {
    pub ek: EncryptionKey<K>,
    pub sk_shares: Vec<VectorShare<K>>, // length n
}

impl Dealer {
    /// Create a dealer where t parties form the active committee and n parties
    /// receive long-term Shamir key shares (any t of which reconstruct sk).
    pub fn new(t: usize, n: usize, p: u64) -> Self {
        assert!(t >= 2, "need t >= 2");
        assert!(t <= n, "need t <= n");
        Dealer { t, n, thr: ThrParams::new(p) }
    }

    /// Run K-PKE key generation and Shamir-share the (NTT-form) secret key
    /// across n parties with polynomial degree t-1.
    pub fn generate_keypair<PARAMS: MlKemParams>(&self) -> KeyShares<{ PARAMS::K }>
    where
        [(); 960 * PARAMS::K + 32]:,
        [(); 1920 * PARAMS::K + 96]:,
        [(); PARAMS::K]:,
        [(); PARAMS::ETA_2]:,
        [(); 64 * PARAMS::ETA_1]:,
        [(); 32 * (PARAMS::D_U * PARAMS::K + PARAMS::D_V)]:,
    {
        let (ek, s): (kpke::KpkeEncryptionKey<{ PARAMS::K }>, Vector<{ PARAMS::K }>) =
            kpke::key_gen::<PARAMS>();

        // Shamir polynomial degree = t - 1, so any t shares reconstruct.
        let sk_shares =
            shamir_poly_ring::share_vector::<{ PARAMS::K }>(&s, self.t - 1, self.n);
        KeyShares { ek, sk_shares }
    }

    /// Generate `ell` independent additive double sharings (<r_j>^{q'}, <r_j>^{mu'})
    /// across the ACTIVE COMMITTEE of size t. Returns a Vec of t DoubleShares,
    /// one per active-committee member (party ids 1..=t).
    ///
    /// Each r_j is sampled fresh with coefficients uniform in [0, mu'), and is
    /// additively shared independently over Z_{q'} and over Z_{mu'} across the
    /// t active members. The ell sharings are packed per-member: member i's
    /// DoubleShare holds Vec-of-arrays for q_prime and mu_prime, both length ell.
    pub fn generate_double_sharing(&self, ell: usize) -> Vec<DoubleShare> {
        assert!(ell >= 1, "ell must be >= 1");
        let mut rng = StdRng::from_entropy();

        // Initialize t empty per-member DoubleShares with capacity ell.
        let mut out: Vec<DoubleShare> = (0..self.t)
            .map(|i| DoubleShare {
                party_id: (i + 1) as u32,
                q_prime: Vec::with_capacity(ell),
                mu_prime: Vec::with_capacity(ell),
            })
            .collect();

        for _ in 0..ell {
            // r_j: 256 coefficients uniform in [0, mu').
            let mut r = [0u64; 256];
            for c in 0..256 {
                r[c] = rng.gen_range(0..self.thr.mu_prime);
            }

            let qp_shares = additive_share_coeffs(&r, self.t, self.thr.q_prime, &mut rng);
            let mp_shares = additive_share_coeffs(&r, self.t, self.thr.mu_prime, &mut rng);

            for i in 0..self.t {
                out[i].q_prime.push(qp_shares[i]);
                out[i].mu_prime.push(mp_shares[i]);
            }
        }

        out
    }
}