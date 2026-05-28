//! Block 5 — Party operations for threshold decryption (Ajax Figure 10).
//!
//! Each party holds:
//!   * its Shamir share of the (NTT-form) secret key:  VectorShare<K>
//!   * its additive double share (<r>^eta, <r>^Phi):    DoubleShare (arrays)
//!
//! The two representations meet here. Step 0 converts the Shamir key share to
//! an ADDITIVE share by multiplying by this party's Lagrange coefficient
//! lambda_j (computed over the active set). After that, all of Step 1's math is
//! on additive shares, matching Fig. 10.
//!
//! Type-flow across the modulus boundary:
//!   <u>^q   : additive share of (Delta*m + e), held as a Ring (coeff form, mod q)
//!   <u'>^eta: modulus-switched, held as [u64;256] (mod eta)  <-- leaves Ring world
//!   <e'>^Phi: <u'>^eta reduced mod Phi, held as [u64;256] (mod Phi)
//!
//! The masking double-shares are [u64;256] arrays (additive, mod eta/Phi), so
//! from Step 1's output onward everything is plain coefficient-vector arithmetic.

use crate::dealer::{DoubleShare, ThrParams};
use crate::params;
use crate::ring::{Ring, RingRepresentation, Vector};
use crate::shamir;
use crate::shamir_ring::VectorShare;

/// A party in the threshold-decryption committee.
pub struct Party<const K: usize> {
    pub id: u32,                 // evaluation point (party id), nonzero
    pub sk_share: VectorShare<K>, // Shamir share of s, NTT form
    pub dbl: DoubleShare,        // additive double share (<r>^eta, <r>^Phi)
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

    /// Step 0 + Step 1 of Fig. 10, combined.
    ///
    /// Inputs:
    ///   * `u`, `v`: the DECOMPRESSED MLWE ciphertext (u in R_q^K, v in R_q),
    ///      both in coefficient form. The caller obtains these by decompressing
    ///      the Cyphertext; see `partial_decrypt_compressed` for the wrapper.
    ///   * `active_ids`: the evaluation points of the t+1 active parties, used
    ///      to compute this party's Lagrange coefficient.
    ///   * `is_v_holder`: exactly one active party adds the public v into its
    ///      share (mirrors the Python `id==0` convention). The driver picks the
    ///      smallest active id as the holder.
    ///
    /// Output: this party's additive shares <u'>^eta and <e'>^Phi, as arrays.
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

        // ----- Step 1a: <u>^q := (v if holder else 0) - <u, s_add> -----------
        // Inner product <u, s_add> over F_q in NTT form, then back to coeffs.
        // u must be in NTT form for Ring::mult; s_add already is.
        let u_ntt = u.clone().ntt();
        let mut inner = u_ntt.inner_product(s_add); // Ring, NTT form
        inner.inverse_ntt();                        // -> coeff form, mod q

        // u_share_q (coeff form, mod q): holder computes v - inner, others -inner.
        let mut u_share_q = Ring::ZEROES_DEGREE255;
        if is_v_holder {
            // v - inner
            for c in 0..256 {
                u_share_q.data[c] =
                    (v.data[c] + params::Q - inner.data[c]) % params::Q;
            }
        } else {
            // -inner  ==  (q - inner) mod q
            for c in 0..256 {
                u_share_q.data[c] = (params::Q - inner.data[c]) % params::Q;
            }
        }

        // ----- Step 1b: modulus switch q -> eta, per coefficient -------------
        // <u'>^eta := round(eta/q * <u>^q) mod eta.  Leaves the Ring world.
        let eta = self.thr.eta;
        let q = self.thr.q;
        let mut u_prime_eta = [0u64; 256];
        for c in 0..256 {
            u_prime_eta[c] = mod_switch_coeff(u_share_q.data[c] as u64, q, eta);
        }

        // ----- Step 1c: <e'>^Phi := <u'>^eta mod Phi ------------------------
        let phi = self.thr.phi;
        let mut e_prime_phi = [0u64; 256];
        for c in 0..256 {
            e_prime_phi[c] = u_prime_eta[c] % phi;
        }

        PartialDecOutput { u_prime_eta, e_prime_phi }
    }

    /// Step 2 (local part): <v>^Phi := <e'>^Phi + <r>^Phi  (mod Phi).
    /// Returns this party's masked share; the driver reconstructs v = e' + r.
    pub fn mask(&self, e_prime_phi: &[u64; 256]) -> [u64; 256] {
        let phi = self.thr.phi;
        let mut masked = [0u64; 256];
        for c in 0..256 {
            masked[c] = (e_prime_phi[c] + self.dbl.phi[c]) % phi;
        }
        masked
    }

    /// Step 3: given the opened v = e' + r (coeff vector, mod Phi but lifted to
    /// Z_eta since coeffs < Phi <= eta), compute:
    ///   <e'>^eta := (v_holder ? v : 0) - <r>^eta        (mod eta)
    ///   <Phi*m>^eta := <u'>^eta - <e'>^eta              (mod eta)
    /// Returns this party's additive share of Phi*m over eta.
    pub fn finalize(
        &self,
        u_prime_eta: &[u64; 256],
        v_opened: &[u64; 256],
        is_v_holder: bool,
    ) -> [u64; 256] {
        let eta = self.thr.eta;

        let mut phi_m = [0u64; 256];
        for c in 0..256 {
            // <e'>^eta : holder uses the public v, others use 0; subtract <r>^eta.
            let e_prime_eta = if is_v_holder {
                (v_opened[c] % eta + eta - self.dbl.eta[c] % eta) % eta
            } else {
                (eta - self.dbl.eta[c] % eta) % eta
            };
            // <Phi*m>^eta := <u'>^eta - <e'>^eta
            phi_m[c] = (u_prime_eta[c] % eta + eta - e_prime_eta) % eta;
        }
        phi_m
    }
}

/// Output of Step 1: this party's additive shares needed for later steps.
#[derive(Clone, Debug)]
pub struct PartialDecOutput {
    pub u_prime_eta: [u64; 256], // <u'>^eta
    pub e_prime_phi: [u64; 256], // <e'>^Phi
}

/// Modulus switch one coefficient from Z_from to Z_to: round(to/from * x) mod to.
/// Done in u128 to avoid overflow (to * x can exceed u64 when both ~2^23..2^53).
fn mod_switch_coeff(x: u64, from: u64, to: u64) -> u64 {
    // round(to * x / from) = (to * x + from/2) / from  (integer division)
    let num = (to as u128) * (x as u128) + (from as u128) / 2;
    ((num / (from as u128)) as u64) % to
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mod_switch_coeff_basic() {
        let q = params::Q64;
        let eta = q - 1;
        // Near-identity since eta ~ q.
        assert_eq!(mod_switch_coeff(0, q, eta), 0);
        assert_eq!(mod_switch_coeff(1, q, eta), 1);
        assert_eq!(mod_switch_coeff(12345, q, eta), 12345);
        // Largest residue wraps just under eta.
        assert_eq!(mod_switch_coeff(q - 1, q, eta), eta - 1);
    }
}