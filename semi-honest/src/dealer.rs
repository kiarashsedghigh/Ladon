//! Block 3 — Trusted Dealer (struct form).
//!
//! Combines the offline responsibilities of the trusted dealer:
//!   1. Run K-PKE key generation.
//!   2. Shamir-share the (NTT-form) secret key across n parties.
//!   3. Generate additive double sharings (<r>^{q'}, <r>^{mu'}) for masking.
//!
//! Mirrors the Python `TrustedDealer`, but:
//!   * secret key is Shamir-shared over F_q (not additive), and kept in NTT
//!     form so the committee can use fast `Ring::mult` during decryption;
//!   * double sharings are ADDITIVE over the two rings Z_{q'} and Z_{mu'},
//!     exactly as in Ajax FDoubleRings (Fig. 11).
//!
//! Notation (matching the paper):
//!   q       : original ciphertext modulus (= params::Q)
//!   q'      : new modulus after switching; chosen as 2^round(log2(q))
//!             so that q' < q and the protocol mod-switches q -> q'.
//!   mu      : q / p   — half-modulus in the original ring (for p = 2)
//!   mu'     : q' / p  — half-modulus in the switched ring (for p = 2)
//!
//! Representation:
//!   * secret-key shares: NTT form (see shamir_ring / dealer keygen).
//!   * double sharings: coefficient form, since r masks the coefficient-form
//!     error e'. These are plain integer-vector shares mod q' / mod mu',
//!     NOT `Ring`s (the `Ring` type is hard-wired to modulus q), so we store
//!     them as [u64; 256] coefficient arrays tagged by their modulus.

use crate::kpke;
use crate::params::*;
use crate::ring::Vector;
use crate::shamir_ring::{self, VectorShare};

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

// ---------------------------------------------------------------------------
// Threshold-decryption parameters (mu, q', mu') derived from q and p.
// ---------------------------------------------------------------------------

/// Parameters for the mask-then-open threshold decryption, derived from the
/// ciphertext modulus q and plaintext modulus p.
///
/// Field names follow the paper:
///   mu       : q / p
///   q_prime  : new modulus after switching (= 2^round(log2(q)); q' < q)
///   mu_prime : q' / p
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

/// One party's additive share of the masking polynomials r_1, ..., r_ell, over
/// BOTH rings. Coefficient form, 256 coefficients each. For majority decoding
/// (Section 5.1 / Algorithm 3 of the paper), the dealer produces `ell` parallel
/// double sharings; the `q_prime`/`mu_prime` Vecs each have length `ell`,
/// where `q_prime[j]` and `mu_prime[j]` are this party's share of r_j over
/// Z_{q'} and Z_{mu'} respectively. Their (coefficient-wise) sums over all
/// parties equal r_j mod q' and mod mu' respectively.
#[derive(Clone, Debug)]
pub struct DoubleShare {
    pub party_id: u32,
    pub q_prime: Vec<[u64; 256]>,  // length ell; shares over Z_{q'}
    pub mu_prime: Vec<[u64; 256]>, // length ell; shares over Z_{mu'}
}

impl DoubleShare {
    /// Number of parallel double sharings this struct carries.
    #[inline]
    pub fn ell(&self) -> usize {
        debug_assert_eq!(self.q_prime.len(), self.mu_prime.len());
        self.q_prime.len()
    }
}

/// Additively share a coefficient vector `secret` (length 256) over modulus m
/// across `n` parties, using the provided RNG. Returns n arrays whose
/// coefficient-wise sum mod m equals `secret`.
fn additive_share_coeffs<R: Rng>(
    secret: &[u64; 256],
    n: usize,
    m: u64,
    rng: &mut R,
) -> Vec<[u64; 256]> {
    let mut shares = vec![[0u64; 256]; n];

    for c in 0..256 {
        let mut acc = 0u64;
        // First n-1 shares uniform in [0, m); last makes the sum = secret.
        for share in shares.iter_mut().take(n - 1) {
            let r = rng.gen_range(0..m);
            share[c] = r;
            acc = (acc + r) % m;
        }
        let last = (secret[c] % m + m - acc) % m;
        shares[n - 1][c] = last;
    }

    shares
}

// ---------------------------------------------------------------------------
// Dealer
// ---------------------------------------------------------------------------

pub type EncryptionKey<const K: usize> = kpke::KpkeEncryptionKey<K>;

/// Trusted dealer holding the threshold configuration.
pub struct Dealer {
    pub t: usize,
    pub n: usize,
    pub thr: ThrParams,
}

/// Output of key generation + secret-key sharing.
pub struct KeyShares<const K: usize> {
    pub ek: EncryptionKey<K>,
    pub sk_shares: Vec<VectorShare<K>>,
}

impl Dealer {
    /// Create a dealer for a (t, n) threshold and plaintext modulus p.
    pub fn new(t: usize, n: usize, p: u64) -> Self {
        assert!(t < n, "need t < n for a (t,n) sharing");
        Dealer { t, n, thr: ThrParams::new(p) }
    }

    /// Run K-PKE key generation and Shamir-share the (NTT-form) secret key.
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

        // Share s directly in NTT form (Shamir is F_q-linear, commutes w/ NTT).
        let sk_shares = shamir_ring::share_vector::<{ PARAMS::K }>(&s, self.t, self.n);

        KeyShares { ek, sk_shares }
    }

    /// Generate `ell` independent additive double sharings (<r_j>^{q'}, <r_j>^{mu'})
    /// for j in 1..=ell, all delivered to the same n parties.
    ///
    /// Each r_j is sampled fresh with coefficients uniform in [0, mu'), and is
    /// additively shared independently over Z_{q'} and over Z_{mu'}. The ell
    /// sharings are packed per-party: party i's returned `DoubleShare` holds
    /// Vec-of-arrays for q_prime and mu_prime, both of length ell.
    ///
    /// `ell` controls the threshold-failure amplification described in
    /// Section 5.1 of the paper: at decoding time the receiver runs the
    /// finalization against each of the ell mask polynomials and takes a
    /// coefficient-wise majority over the ell candidate plaintexts.
    pub fn generate_double_sharing(&self, ell: usize) -> Vec<DoubleShare> {
        assert!(ell >= 1, "ell must be >= 1");
        let mut rng = StdRng::from_entropy();

        // Initialize n empty per-party DoubleShares with capacity ell.
        let mut out: Vec<DoubleShare> = (0..self.n)
            .map(|i| DoubleShare {
                party_id: (i + 1) as u32,
                q_prime: Vec::with_capacity(ell),
                mu_prime: Vec::with_capacity(ell),
            })
            .collect();

        // Generate ell independent double sharings, appending each round's
        // per-party shares to the corresponding DoubleShare.
        for _ in 0..ell {
            // r_j: 256 coefficients uniform in [0, mu').
            let mut r = [0u64; 256];
            for c in 0..256 {
                r[c] = rng.gen_range(0..self.thr.mu_prime);
            }

            let qp_shares = additive_share_coeffs(&r, self.n, self.thr.q_prime, &mut rng);
            let mp_shares = additive_share_coeffs(&r, self.n, self.thr.mu_prime, &mut rng);

            for i in 0..self.n {
                out[i].q_prime.push(qp_shares[i]);
                out[i].mu_prime.push(mp_shares[i]);
            }
        }

        out
    }
}