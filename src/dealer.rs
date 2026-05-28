//! Block 3 — Trusted Dealer (struct form).
//!
//! Combines the offline responsibilities of the trusted dealer:
//!   1. Run K-PKE key generation.
//!   2. Shamir-share the (NTT-form) secret key across n parties.
//!   3. Generate additive double sharings (<r>^eta, <r>^Phi) for masking.
//!
//! Mirrors the Python `TrustedDealer`, but:
//!   * secret key is Shamir-shared over F_q (not additive), and kept in NTT
//!     form so the committee can use fast `Ring::mult` during decryption;
//!   * double sharings are ADDITIVE over the two rings Z_eta and Z_Phi, exactly
//!     as in Ajax FDoubleRings (Fig. 11). Per your choice, eta = q - 1.
//!
//! Representation:
//!   * secret-key shares: NTT form (see shamir_ring / dealer keygen).
//!   * double sharings: coefficient form, since r masks the coefficient-form
//!     noise e'. These are plain integer-vector shares mod eta / mod Phi, NOT
//!     `Ring`s (the `Ring` type is hard-wired to modulus q), so we store them
//!     as [u64; 256] coefficient arrays tagged by their modulus.

use crate::kpke;
use crate::params::*;
use crate::ring::Vector;
use crate::shamir_ring::{self, VectorShare};

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

// ---------------------------------------------------------------------------
// Threshold-decryption parameters (eta, Phi, Delta) derived from q and p.
// ---------------------------------------------------------------------------

/// Parameters for the mask-then-open threshold decryption, derived from the
/// ciphertext modulus q and plaintext modulus p.
#[derive(Clone, Copy, Debug)]
pub struct ThrParams {
    pub q: u64,
    pub p: u64,
    pub delta: u64, // floor(q / p)
    pub eta: u64,   // modulus-switch target; per user choice eta = q - 1
    pub phi: u64,   // eta / p
}

impl ThrParams {
    /// Build from q (= params::Q) and a power-of-two plaintext modulus p.
    /// Uses eta = q - 1 (user's choice). Requires p | eta.
    pub fn new(p: u64) -> Self {
        assert!(p.is_power_of_two(), "p must be a power of two");
        let q = Q64;
        let eta = q - 1;
        assert!(eta % p == 0, "p must divide eta = q-1");
        ThrParams {
            q,
            p,
            delta: q / p,
            eta,
            phi: eta / p,
        }
    }
}

// ---------------------------------------------------------------------------
// Additive double sharing over two rings (Z_eta, Z_Phi).
// ---------------------------------------------------------------------------

/// One party's additive share of the masking polynomial r, over BOTH rings.
/// Coefficient form, 256 coefficients. `eta`/`phi` arrays hold the per-modulus
/// shares; their (coefficient-wise) sums over all parties equal r mod eta and
/// r mod Phi respectively.
#[derive(Clone, Debug)]
pub struct DoubleShare {
    pub party_id: u32,
    pub eta: [u64; 256], // share over Z_eta
    pub phi: [u64; 256], // share over Z_Phi
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
        [(); 384 * PARAMS::K + 32]:,
        [(); 768 * PARAMS::K + 96]:,
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

    /// Generate one additive double sharing (<r>^eta, <r>^Phi) for n parties.
    ///
    /// Samples a random masking polynomial r with coefficients uniform in
    /// [0, Phi), then additively shares it over Z_eta and over Z_Phi. Because
    /// every coefficient of r is < Phi <= eta, the same integer r is consistent
    /// across both rings (no wraparound when viewed in Z_eta).
    pub fn generate_double_sharing(&self) -> Vec<DoubleShare> {
        let mut rng = StdRng::from_entropy();

        // r: 256 coefficients uniform in [0, Phi).
        let mut r = [0u64; 256];
        for c in 0..256 {
            r[c] = rng.gen_range(0..self.thr.phi);
        }

        // Additive shares over each ring.
        let eta_shares = additive_share_coeffs(&r, self.n, self.thr.eta, &mut rng);
        let phi_shares = additive_share_coeffs(&r, self.n, self.thr.phi, &mut rng);

        (0..self.n)
            .map(|i| DoubleShare {
                party_id: (i + 1) as u32,
                eta: eta_shares[i],
                phi: phi_shares[i],
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ring::RingRepresentation;
    use crate::shamir_ring::reconstruct_vector;

    #[test]
    fn test_secret_key_shares_reconstruct() {
        // Deal a key, then verify any t+1 shares reconstruct the SAME key, and
        // that the reconstructed key is in NTT form (so Ring::mult will work).
        type P = MlKem512;
        let dealer = Dealer::new(2, 5, 2);
        let ks = dealer.generate_keypair::<P>();
        assert_eq!(ks.sk_shares.len(), 5);

        let subset_a: Vec<_> = ks.sk_shares[0..3].to_vec(); // ids 1,2,3
        let subset_b: Vec<_> = ks.sk_shares[2..5].to_vec(); // ids 3,4,5

        let rec_a = reconstruct_vector(&subset_a);
        let rec_b = reconstruct_vector(&subset_b);

        assert_eq!(rec_a, rec_b, "different quorums must reconstruct the same key");
        for ring in rec_a.data.iter() {
            assert_eq!(ring.t, RingRepresentation::NTT);
        }
    }

    #[test]
    fn test_double_sharing_reconstructs_same_r() {
        // The eta-shares and the Phi-shares must reconstruct to the SAME r
        // (mod Phi), confirming the double sharing is consistent across rings.
        let dealer = Dealer::new(2, 5, 2);
        let ds = dealer.generate_double_sharing();
        assert_eq!(ds.len(), 5);

        let phi = dealer.thr.phi;
        let eta = dealer.thr.eta;

        for c in 0..256 {
            // Sum eta-shares mod eta, then reduce mod Phi.
            let mut r_from_eta = 0u64;
            for s in &ds {
                r_from_eta = (r_from_eta + s.eta[c]) % eta;
            }
            let r_from_eta_mod_phi = r_from_eta % phi;

            // Sum phi-shares mod Phi.
            let mut r_from_phi = 0u64;
            for s in &ds {
                r_from_phi = (r_from_phi + s.phi[c]) % phi;
            }

            assert_eq!(
                r_from_eta_mod_phi, r_from_phi,
                "coeff {c}: r mod Phi disagrees between the two ring sharings"
            );
        }
    }
}