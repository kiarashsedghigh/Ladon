//! Block 5 — Party operations for threshold decryption.
//!
//! Each party holds:
//!   * its Shamir share of the (NTT-form) secret key:  VectorShare<K>
//!   * its additive double share (<r>^{q'}, <r>^{mu'}): DoubleShare (arrays)
//!
//! Protocol notation (matches the paper):
//!   q       : original ciphertext modulus
//!   q'      : new modulus after switching (= 2^round(log2(q)))
//!   mu      : q / p   (half-modulus in the original ring for p = 2)
//!   mu'     : q' / p  (half-modulus in the switched ring for p = 2)
//!   w       : v - <sk, u>     (result of linear decryption, mod q)
//!   w'      : mod-switch of w to q'   (share of mu' * m + e' mod q')
//!   e'      : w' mod mu'             (share of the accumulated error alone)
//!
//! Step 0 converts the Shamir key share to an ADDITIVE share by multiplying
//! by this party's Lagrange coefficient lambda_j (computed over the active
//! set). After that, all of the protocol math is on additive shares.
//!
//! Type-flow across the modulus boundary:
//!   <w>^q   : additive share of (mu*m + e), held as a Ring (coeff form, mod q)
//!   <w'>^{q'}: modulus-switched, held as [u64;256] (mod q')  <- leaves Ring world
//!   <e'>^{mu'}: <w'>^{q'} reduced mod mu', held as [u64;256] (mod mu')
//!
//! The masking double-shares are [u64;256] arrays (additive, mod q'/mu'), so
//! from Step 1's output onward everything is plain coefficient-vector arithmetic.

use crate::dealer::{DoubleShare, ThrParams};
use crate::params;
use crate::ring::{Ring, RingRepresentation, Vector};
use crate::shamir;
use crate::shamir_poly_ring::VectorShare;

/// A party in the threshold-decryption committee.
pub struct Party<const K: usize> {
    pub id: u32,                  // evaluation point (party id), nonzero
    pub sk_share: VectorShare<K>, // Shamir share of s, NTT form
    pub dbl: DoubleShare,         // additive double share (<r>^{q'}, <r>^{mu'})
    pub thr: ThrParams,
}

impl<const K: usize> Party<K> {
    pub fn new(
        id: u32,
        sk_share: VectorShare<K>,
        dbl: DoubleShare,
        thr: ThrParams,
    ) -> Self {
        debug_assert_eq!(id, sk_share.x, "party id must match its share's eval point");
        debug_assert_eq!(id, dbl.party_id, "party id must match its double-share id");
        Party { id, sk_share, dbl, thr }
    }

    /// Wrapper: take the COMPRESSED ciphertext and decompress internally, then
    /// run Step 0 + Step 1. This is the entry point matching "party takes the
    /// compressed Cyphertext". D_U/D_V are the ciphertext compression widths.
    pub fn partial_decrypt_compressed<const D_U: usize, const D_V: usize>(
        &self,
        c: crate::kpke::Cyphertext<K, D_U, D_V>,
        active_ids: &[u32],
        is_v_holder: bool,
    ) -> PartialDecOutput {
        // c.0: Compressed<D_U, Vector<K>>  ->  u : Vector<K> (coeff form)
        // c.1: Compressed<D_V, Ring>       ->  v : Ring      (coeff form)
        let u = c.0.decompress();
        let v = c.1.decompress();
        self.partial_decrypt(&u, &v, active_ids, is_v_holder)
    }

    /// Steps 1-3 of the partial decryption protocol, combined.
    ///   Step 1 (linear decryption): w := v - <sk, u>  (mod q)
    ///   Step 2 (modulus switch):    w' := round((q'/q) * w)  (mod q')
    ///   Step 3 (isolate error):     e' := w' mod mu'
    ///
    /// Inputs:
    ///   * `u`, `v`: the DECOMPRESSED MLWE ciphertext (u in R_q^K, v in R_q),
    ///      both in coefficient form. The caller obtains these by decompressing
    ///      the Cyphertext; see `partial_decrypt_compressed` for the wrapper.
    ///   * `active_ids`: the evaluation points of the t active parties, used
    ///      to compute this party's Lagrange coefficient.
    ///   * `is_v_holder`: exactly one active party adds the public v into its
    ///      share (mirrors the Python `id==0` convention). The driver picks the
    ///      smallest active id as the holder.
    ///
    /// Output: this party's additive shares <w'>^{q'} and <e'>^{mu'}, as arrays.
    pub fn partial_decrypt(
        &self,
        u: &Vector<K>,
        v: &Ring,
        active_ids: &[u32],
        is_v_holder: bool,
    ) -> PartialDecOutput {
        // ----- Step 0: Shamir -> additive via local Lagrange -----------------
        // lambda_j scales this party's WHOLE key share (F_q-linear, NTT-safe).
        let lambda = shamir::lagrange_coeff_at_zero(active_ids, self.id);

        // additive share of s_j = lambda_j * [[s]]_j  (still NTT form).
        let mut s_add: Vector<K> = self.sk_share.vector.clone();
        for k in 0..K {
            s_add.data[k].scalar_mul(lambda); // mod q, NTT-domain scalar mul
        }

        // ----- Step 1: <w>^q := (v if holder else 0) - <u, s_add>  (mod q) --
        // Inner product <u, s_add> over F_q in NTT form, then back to coeffs.
        // u must be in NTT form for Ring::mult; s_add already is.
        let u_ntt = u.clone().ntt();
        let mut inner = u_ntt.inner_product(s_add); // Ring, NTT form
        inner.inverse_ntt();                        // -> coeff form, mod q

        // w_share_q (coeff form, mod q): holder computes v - inner, others -inner.
        let mut w_share_q = Ring::ZEROES_DEGREE255;
        if is_v_holder {
            // v - inner
            for c in 0..256 {
                w_share_q.data[c] =
                    (v.data[c] + params::Q - inner.data[c]) % params::Q;
            }
        } else {
            // -inner  ==  (q - inner) mod q
            for c in 0..256 {
                w_share_q.data[c] = (params::Q - inner.data[c]) % params::Q;
            }
        }

        // ----- Step 2: modulus switch q -> q', per coefficient ---------------
        // <w'>^{q'} := round(q'/q * <w>^q) mod q'.  Leaves the Ring world.
        let q_prime = self.thr.q_prime;
        let q = self.thr.q;
        let mut w_prime = [0u64; 256];
        for c in 0..256 {
            w_prime[c] = mod_switch_coeff(w_share_q.data[c] as u64, q, q_prime);
        }

        // ----- Step 3: <e'>^{mu'} := <w'>^{q'} mod mu' ----------------------
        let mu_prime = self.thr.mu_prime;
        let mut e_prime = [0u64; 256];
        for c in 0..256 {
            e_prime[c] = w_prime[c] % mu_prime;
        }

        PartialDecOutput { w_prime, e_prime }
    }

    /// Step 4 (local part), batched over ell: for each j in 0..ell, compute
    /// <e_tilde_j>^{mu'} := <e'>^{mu'} + <d_j>^{mu'}  (mod mu').
    ///
    /// Returns a Vec of length ell. The same e'-share (produced by Step 3)
    /// is masked against each of the ell preprocessed double sharings.
    pub fn mask(&self, e_prime: &[u64; 256]) -> Vec<[u64; 256]> {
        let mu_prime = self.thr.mu_prime;
        let ell = self.dbl.ell();
        let mut out = Vec::with_capacity(ell);
        for j in 0..ell {
            let mut masked = [0u64; 256];
            for c in 0..256 {
                masked[c] = (e_prime[c] + self.dbl.mu_prime[j][c]) % mu_prime;
            }
            out.push(masked);
        }
        out
    }

    /// Step 5 batched over ell: given the ell opened e_tilde_j = e' + d_j
    /// vectors (one per parallel double sharing), compute this party's
    /// additive share of mu'*m once per j:
    ///   <e'>^{q'}_j      := (v_holder ? e_tilde_j : 0) - <d_j>^{q'}  (mod q')
    ///   <mu'*m>^{q'}_j   := <w'>^{q'} - <e'>^{q'}_j                  (mod q')
    ///
    /// Returns a Vec of length ell. <w'>^{q'} is shared across all ell
    /// candidates (it depends only on the ciphertext + key share), so the
    /// only ell-varying inputs are the opened e_tilde_j and this party's
    /// d_j share over q' for that round.
    pub fn finalize(
        &self,
        w_prime: &[u64; 256],
        e_tilde_opened: &[[u64; 256]],
        is_v_holder: bool,
    ) -> Vec<[u64; 256]> {
        let q_prime = self.thr.q_prime;
        let ell = self.dbl.ell();
        assert_eq!(
            e_tilde_opened.len(),
            ell,
            "finalize: got {} opened e_tilde's but party holds ell = {}",
            e_tilde_opened.len(),
            ell
        );

        let mut out = Vec::with_capacity(ell);
        for j in 0..ell {
            let e_tilde_j = &e_tilde_opened[j];
            let d_qprime_j = &self.dbl.q_prime[j];

            let mut mu_m = [0u64; 256];
            for c in 0..256 {
                let e_prime_qprime = if is_v_holder {
                    (e_tilde_j[c] % q_prime + q_prime - d_qprime_j[c] % q_prime) % q_prime
                } else {
                    (q_prime - d_qprime_j[c] % q_prime) % q_prime
                };
                mu_m[c] = (w_prime[c] % q_prime + q_prime - e_prime_qprime) % q_prime;
            }
            out.push(mu_m);
        }
        out
    }
}

/// Output of Steps 1-3: this party's additive shares needed for later steps.
///   w_prime : <w'>^{q'}   (share of mu' * m + e' over Z_{q'})
///   e_prime : <e'>^{mu'}  (share of accumulated error e' over Z_{mu'})
#[derive(Clone, Debug)]
pub struct PartialDecOutput {
    pub w_prime: [u64; 256], // <w'>^{q'}
    pub e_prime: [u64; 256], // <e'>^{mu'}
}

/// Modulus switch one coefficient from Z_from to Z_to: round(to/from * x) mod to.
/// Done in u128 to avoid overflow.
fn mod_switch_coeff(x: u64, from: u64, to: u64) -> u64 {
    // round(to * x / from) = (to * x + from/2) / from  (integer division)
    let num = (to as u128) * (x as u128) + (from as u128) / 2;
    ((num / (from as u128)) as u64) % to
}