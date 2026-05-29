//! Block 4 — Party operations for SPDZ2k threshold decryption.
//!
//! Each party holds:
//!   * its LIFTED additive share of the secret key:  VectorShare128<K>
//!     (coefficients in [0, 2^(k+s)), one polynomial per module slot)
//!   * its additive double share of the masking polynomial r:  DoubleShare
//!     (r_q_share over q = 2^k, r_delta_share over Delta = q/p)
//!
//! Differs from the Shamir/prime party (party.rs):
//!   * NO Lagrange conversion — shares are ALREADY additive end to end.
//!   * NO modulus switch — q = 2^k has p | q, so Delta = q/p is exact and the
//!     mask-then-open math lives in (q, Delta) directly.
//!   * Inner product <u, s_share> is computed in the LIFTED ring 2^(k+s) and
//!     reduced to q at the very end (the lift commutes with mod-q reduction).
//!
//! Type-flow across the protocol:
//!   sk_share         : u128 coefficients in [0, 2^(k+s))            (lifted)
//!   <u,s>^(i) accum  : i128 (need ~91 bits magnitude; signed for v - inner)
//!   <u>^q after Step1: u128 reduced into [0, q)                     (base)
//!   <e>^Delta share  : u128 in [0, Delta)                           (base)
//!   <Phi*m>^q share  : u128 in [0, q)                               (base)

use crate::additive_2k::SpdzParams;
use crate::additive_ring::VectorShare128;
use crate::dealer_spdz::DoubleShare;
use crate::negacyclic;
use crate::ring::{Ring, Vector};

/// A party in the dishonest-majority committee.
pub struct PartySpdz<const K: usize> {
    pub id: u32,
    pub sk_share: VectorShare128<K>,
    pub dbl: DoubleShare,
    pub params: SpdzParams,
}

impl<const K: usize> PartySpdz<K> {
    pub fn new(
        id: u32,
        sk_share: VectorShare128<K>,
        dbl: DoubleShare,
        params: SpdzParams,
    ) -> Self {
        debug_assert_eq!(id, sk_share.x, "party id must match its key-share id");
        debug_assert_eq!(id, dbl.party_id, "party id must match its double-share id");
        PartySpdz { id, sk_share, dbl, params }
    }

    /// Step 1 of the protocol (combined 1a + 1b + 1c).
    ///
    /// Inputs:
    ///   * `u`, `v` — the DECOMPRESSED ciphertext: u in R_q^K, v in R_q,
    ///     both coefficient form (kpke::decrypt_2k decompresses internally;
    ///     here the caller does the decompression).
    ///   * `is_v_holder` — exactly one party adds the public v into its share.
    ///     The driver designates the smallest id as the holder.
    ///
    /// Output: this party's additive shares <u>^q and <e>^Delta (arrays).
    pub fn partial_decrypt(
        &self,
        u: &Vector<K>,
        v: &Ring,
        is_v_holder: bool,
    ) -> PartialDecOutput {
        let q = self.params.q;
        let delta = self.params.delta;

        // ---- Step 1a: <u, s_share> via negacyclic poly product -------------
        // For each output coefficient c in 0..256:
        //   inner[c] = sum over j in 0..K of (u_j * s_share_j)[c]
        // where (u_j * s_share_j)[c] is the negacyclic poly product coefficient:
        //   sum_{a+b=c}     u_j[a] * s_j[b]
        // - sum_{a+b=c+256} u_j[a] * s_j[b]    (X^256 = -1)
        //
        // u_j coeffs are u32 in [0, q); s_share_j coeffs are u128 in [0, 2^(k+s)).
        // Worst-case magnitude per c: ~ N * K * q * 2^(k+s) ~ 2^91 at (k,s,K)=(20,40,4).
        // That fits i128 (~127 bits signed) comfortably; no intermediate
        // reductions needed inside the K-loop.
        let mut inner = [0i128; 256];
        for j in 0..K {
            let s_j_coeffs: [u128; 256] = self.sk_share.rings[j].data;
            let u_j_coeffs: [u32; 256] = u.data[j].data;
            for c in 0..256 {
                // a + b = c
                for a in 0..=c {
                    let b = c - a;
                    inner[c] += (u_j_coeffs[a] as i128) * (s_j_coeffs[b] as i128);
                }
                // a + b = c + 256 (wraparound, negated)
                for a in (c + 1)..256 {
                    let b = c + 256 - a;
                    inner[c] -= (u_j_coeffs[a] as i128) * (s_j_coeffs[b] as i128);
                }
            }
        }

        // ---- Step 1a (cont): <u>^q := (holder? v : 0) - inner, mod q -------
        // The lift commutes with mod-q reduction: reducing each party's share
        // mod q first and summing gives the same result as summing in the
        // lifted ring then reducing. We reduce here.
        let mut u_share_q = [0u128; 256];
        for c in 0..256 {
            let inner_mod = inner[c].rem_euclid(q as i128) as u128;
            let v_term: u128 = if is_v_holder { v.data[c] as u128 } else { 0 };
            // (v_term - inner_mod) mod q, both already in [0, q).
            u_share_q[c] = if v_term >= inner_mod {
                v_term - inner_mod
            } else {
                q + v_term - inner_mod
            };
        }

        // ---- Step 1c: <e>^Delta := <u>^q mod Delta (per coefficient) -------
        let mut e_share_delta = [0u128; 256];
        for c in 0..256 {
            e_share_delta[c] = u_share_q[c] % delta;
        }

        PartialDecOutput { u_share_q, e_share_delta }
    }

    /// Step 2 (local part): <v_masked>^Delta := <e>^Delta + <r>^Delta (mod Delta).
    /// The driver then sums masked shares to open v = e + r.
    pub fn mask(&self, e_share_delta: &[u128; 256]) -> [u128; 256] {
        let delta = self.params.delta;
        let mut masked = [0u128; 256];
        for c in 0..256 {
            masked[c] = (e_share_delta[c] + self.dbl.r_delta_share[c]) % delta;
        }
        masked
    }

    /// Step 3 (after v = e + r is opened): compute this party's share of
    /// <Delta * m>^q.
    ///
    /// Steps in one shot:
    ///   <e>^q := (holder? v_opened : 0) - <r>^q              (mod q)
    ///   <Delta*m>^q := <u>^q - <e>^q                          (mod q)
    pub fn finalize(
        &self,
        u_share_q: &[u128; 256],
        v_opened: &[u128; 256],
        is_v_holder: bool,
    ) -> [u128; 256] {
        let q = self.params.q;
        let mut delta_m = [0u128; 256];
        for c in 0..256 {
            // <e>^q reconstruction: v_holder uses public v; subtract <r>^q.
            let v_term: u128 = if is_v_holder { v_opened[c] % q } else { 0 };
            let r_q = self.dbl.r_q_share[c] % q;
            let e_q = if v_term >= r_q {
                v_term - r_q
            } else {
                q + v_term - r_q
            };
            // <Delta*m>^q = <u>^q - <e>^q
            let ushare = u_share_q[c] % q;
            delta_m[c] = if ushare >= e_q { ushare - e_q } else { q + ushare - e_q };
        }
        delta_m
    }
}

/// Output of Step 1: the two arrays needed for the rest of the protocol.
#[derive(Clone, Debug)]
pub struct PartialDecOutput {
    pub u_share_q: [u128; 256],     // <u>^q in [0, q)
    pub e_share_delta: [u128; 256], // <e>^Delta in [0, Delta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::additive_ring;
    use crate::dealer_spdz::DealerSpdz;
    use crate::params::*;

    /// Sanity: with the SECRET key reconstructed (not threshold) and a known
    /// ciphertext, partial_decrypt outputs that sum to the centralized
    /// decryption value (Delta*m + e mod q). We don't need a real ciphertext;
    /// we just verify that summing all parties' partial_decrypt outputs equals
    /// what a single-party "v - <u,s>" would produce.
    #[test]
    fn test_partial_decrypt_shares_sum_to_v_minus_us() {
        type P = MlKem512;
        const K: usize = <P as MlKemParams>::K;
        let n = 3;
        let dealer = DealerSpdz::new(n, 20, 40, 2);

        // Use the dealer to produce a real secret-key sharing.
        let ks = dealer.generate_keypair::<P>();
        let dbl = dealer.generate_double_sharing();
        let s_full = additive_ring::reconstruct_vector_lifted(&ks.sk_shares, &dealer.params);

        // Make up a coefficient-form ciphertext (u, v) of the right shape.
        // u: K rings with u32 coeffs in [0, q); v: one ring same range.
        let q = dealer.params.q as u32;
        let mut u = Vector::<K>::new_degree255();
        for j in 0..K {
            for c in 0..256 {
                u.data[j].data[c] = ((j as u32 * 31 + c as u32 * 7) ^ 0xABCD) % q;
            }
        }
        let mut v = Ring::ZEROES_DEGREE255;
        for c in 0..256 {
            v.data[c] = (c as u32 * 13 + 11) % q;
        }

        // Build parties and run partial_decrypt for each.
        let parties: Vec<PartySpdz<K>> = ks
            .sk_shares
            .iter()
            .zip(dbl.iter())
            .map(|(sk, d)| PartySpdz::new(sk.x, sk.clone(), d.clone(), dealer.params))
            .collect();
        let min_id = parties.iter().map(|p| p.id).min().unwrap();

        let outs: Vec<_> = parties
            .iter()
            .map(|p| p.partial_decrypt(&u, &v, p.id == min_id))
            .collect();

        // Sum parties' u_share_q mod q -> should equal (v - <u, s_full>) mod q.
        // Compute reference: <u, s_full> by direct negacyclic poly multiply +
        // sum across K, then v - inner mod q.
        let mut reference = [0i128; 256];
        for j in 0..K {
            let s_j = s_full.data[j].data; // u32 in [0, q)
            let u_j = u.data[j].data;
            for c in 0..256 {
                let mut acc: i128 = 0;
                for a in 0..=c {
                    acc += (u_j[a] as i128) * (s_j[c - a] as i128);
                }
                for a in (c + 1)..256 {
                    acc -= (u_j[a] as i128) * (s_j[c + 256 - a] as i128);
                }
                reference[c] += acc;
            }
        }
        let q_i = dealer.params.q as i128;
        let mut expected = [0u128; 256];
        for c in 0..256 {
            let inner_mod = reference[c].rem_euclid(q_i) as u128;
            let v_term = v.data[c] as u128;
            expected[c] = if v_term >= inner_mod {
                v_term - inner_mod
            } else {
                dealer.params.q + v_term - inner_mod
            };
        }

        // Sum of parties' u_share_q mod q must equal expected per coefficient.
        for c in 0..256 {
            let s = outs
                .iter()
                .fold(0u128, |acc, o| (acc + o.u_share_q[c]) % dealer.params.q);
            assert_eq!(s, expected[c], "u_share_q reconstruction mismatch at coeff {c}");
        }
    }

    /// Mask-and-open consistency: sum of masked shares mod Delta equals
    /// (sum of e shares mod Delta) + (sum of r_delta shares mod Delta), all mod Delta.
    #[test]
    fn test_mask_open_consistency() {
        type P = MlKem512;
        const K: usize = <P as MlKemParams>::K;
        let dealer = DealerSpdz::new(3, 20, 40, 2);
        let ks = dealer.generate_keypair::<P>();
        let dbl = dealer.generate_double_sharing();

        // Fake e shares directly (each party picks a random e_share_delta).
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let parties: Vec<PartySpdz<K>> = ks
            .sk_shares
            .iter()
            .zip(dbl.iter())
            .map(|(sk, d)| PartySpdz::new(sk.x, sk.clone(), d.clone(), dealer.params))
            .collect();

        let e_shares: Vec<[u128; 256]> = (0..parties.len())
            .map(|_| {
                let mut arr = [0u128; 256];
                for c in 0..256 {
                    arr[c] = rng.gen_range(0..dealer.params.delta);
                }
                arr
            })
            .collect();

        let masked: Vec<[u128; 256]> = parties
            .iter()
            .zip(e_shares.iter())
            .map(|(p, e)| p.mask(e))
            .collect();

        let delta = dealer.params.delta;
        for c in 0..256 {
            let lhs = masked
                .iter()
                .fold(0u128, |acc, m| (acc + m[c]) % delta);
            let sum_e = e_shares
                .iter()
                .fold(0u128, |acc, e| (acc + e[c]) % delta);
            let sum_r = dbl
                .iter()
                .fold(0u128, |acc, d| (acc + d.r_delta_share[c]) % delta);
            assert_eq!(lhs, (sum_e + sum_r) % delta);
        }
    }
}