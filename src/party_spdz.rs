//! Block 4 — Party operations for SPDZ2k threshold decryption.
//!
//! Each party holds:
//!   * its LIFTED additive share of the secret key:  VectorShare128<K>
//!     (coefficients in [0, 2^(k+s)), one polynomial per module slot)
//!   * its additive double share of the ell masking polynomials d_1,..,d_ell:
//!     DoubleShare (q_prime[j] over Z_{q'}, mu_prime[j] over Z_{mu'})
//!
//! Protocol notation (matches the paper):
//!   q       : PKE modulus (= 2^k)
//!   q'      : modulus after switching (here q' = q since q is power of 2)
//!   mu      : q / p
//!   mu'     : q' / p (= mu here)
//!   w       : v - <sk, u>     (result of linear decryption, mod q)
//!   w'      : mod-switch of w to q'   (share of mu' * m + e' mod q')
//!   e'      : w' mod mu'             (share of accumulated error alone)
//!
//! Differs from the Shamir party:
//!   * NO Lagrange — shares are ALREADY additive end to end.
//!   * Mod switch is identity here (q' = q), but the variable carrying the
//!     value uses paper notation `w_prime` for clarity / symmetry.
//!   * Inner product <u, s_share> is computed in the LIFTED ring 2^(k+s) and
//!     reduced to q at the very end.

use crate::additive_2k::SpdzParams;
use crate::additive_ring::VectorShare128;
use crate::dealer_spdz::DoubleShare;
use crate::ring::{Ring, Vector};

/// A party in the dishonest-majority committee.
pub struct PartySpdz<const K: usize> {
    pub id: u32,
    pub sk_share: VectorShare128<K>,
    pub dbl: DoubleShare,
    pub thr: SpdzParams,
}

impl<const K: usize> PartySpdz<K> {
    pub fn new(
        id: u32,
        sk_share: VectorShare128<K>,
        dbl: DoubleShare,
        thr: SpdzParams,
    ) -> Self {
        debug_assert_eq!(id, sk_share.x, "party id must match its key-share id");
        debug_assert_eq!(id, dbl.party_id, "party id must match its double-share id");
        PartySpdz { id, sk_share, dbl, thr }
    }

    /// Steps 1-3 of the partial decryption protocol, combined.
    ///   Step 1 (linear decryption): w := v - <sk, u>  (mod q)
    ///   Step 2 (modulus switch):    w' := round((q'/q) * w)  (mod q'); identity here
    ///   Step 3 (isolate error):     e' := w' mod mu'
    ///
    /// Inputs:
    ///   * `u`, `v` — DECOMPRESSED ciphertext (u in R_q^K, v in R_q), coeff form.
    ///   * `is_v_holder` — exactly one party adds the public v into its share.
    ///
    /// Output: this party's additive shares <w'>^{q'} and <e'>^{mu'}.
    pub fn partial_decrypt(
        &self,
        u: &Vector<K>,
        v: &Ring,
        is_v_holder: bool,
    ) -> PartialDecOutput {
        let q = self.thr.q;
        let q_prime = self.thr.q_prime;
        let mu_prime = self.thr.mu_prime;

        // ---- Step 1: <w>^q := (holder ? v : 0) - <u, sk> ; via negacyclic ---
        // inner[c] = sum_j (u_j * s_j)[c]  (negacyclic poly product, mod nothing yet)
        // Magnitude ~ N * K * q * 2^(k+s) ~ 2^91 at (k,s,K)=(20,40,4), fits i128.
        let mut inner = [0i128; 256];
        for j in 0..K {
            let s_j_coeffs: [u128; 256] = self.sk_share.rings[j].data;
            let u_j_coeffs: [u32; 256] = u.data[j].data;
            for c in 0..256 {
                for a in 0..=c {
                    let b = c - a;
                    inner[c] += (u_j_coeffs[a] as i128) * (s_j_coeffs[b] as i128);
                }
                for a in (c + 1)..256 {
                    let b = c + 256 - a;
                    inner[c] -= (u_j_coeffs[a] as i128) * (s_j_coeffs[b] as i128);
                }
            }
        }

        // <w>^q (additive share, mod q). The lift commutes with mod-q reduction.
        let mut w_share_q = [0u128; 256];
        for c in 0..256 {
            let inner_mod = inner[c].rem_euclid(q as i128) as u128;
            let v_term: u128 = if is_v_holder { v.data[c] as u128 } else { 0 };
            w_share_q[c] = if v_term >= inner_mod {
                v_term - inner_mod
            } else {
                q + v_term - inner_mod
            };
        }

        // ---- Step 2: modulus switch q -> q' ; identity here since q' = q. ---
        let mut w_prime = [0u128; 256];
        for c in 0..256 {
            w_prime[c] = mod_switch_coeff(w_share_q[c], q, q_prime);
        }

        // ---- Step 3: <e'>^{mu'} := <w'>^{q'} mod mu' ------------------------
        let mut e_prime = [0u128; 256];
        for c in 0..256 {
            e_prime[c] = w_prime[c] % mu_prime;
        }

        PartialDecOutput { w_prime, e_prime }
    }

    /// Step 4 (local part), batched over ell: for j in 0..ell,
    ///   <e_tilde_j>^{mu'} := <e'>^{mu'} + <d_j>^{mu'}  (mod mu').
    pub fn mask(&self, e_prime: &[u128; 256]) -> Vec<[u128; 256]> {
        let mu_prime = self.thr.mu_prime;
        let ell = self.dbl.ell();
        let mut out = Vec::with_capacity(ell);
        for j in 0..ell {
            let mut masked = [0u128; 256];
            for c in 0..256 {
                masked[c] = (e_prime[c] + self.dbl.mu_prime[j][c]) % mu_prime;
            }
            out.push(masked);
        }
        out
    }

    /// Step 5 batched over ell: given the ell opened e_tilde_j vectors,
    /// compute this party's additive share of mu'*m once per j:
    ///   <e'>^{q'}_j     := (v_holder ? e_tilde_j : 0) - <d_j>^{q'}  (mod q')
    ///   <mu'*m>^{q'}_j  := <w'>^{q'} - <e'>^{q'}_j                  (mod q')
    pub fn finalize(
        &self,
        w_prime: &[u128; 256],
        e_tilde_opened: &[[u128; 256]],
        is_v_holder: bool,
    ) -> Vec<[u128; 256]> {
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

            let mut mu_m = [0u128; 256];
            for c in 0..256 {
                let d_qp = d_qprime_j[c] % q_prime;
                let e_prime_qprime = if is_v_holder {
                    let e_t = e_tilde_j[c] % q_prime;
                    if e_t >= d_qp { e_t - d_qp } else { q_prime + e_t - d_qp }
                } else {
                    (q_prime - d_qp) % q_prime
                };
                let w_p = w_prime[c] % q_prime;
                mu_m[c] = if w_p >= e_prime_qprime {
                    w_p - e_prime_qprime
                } else {
                    q_prime + w_p - e_prime_qprime
                };
            }
            out.push(mu_m);
        }
        out
    }
}

/// Output of Steps 1-3.
///   w_prime : <w'>^{q'}   (share of mu' * m + e' over Z_{q'})
///   e_prime : <e'>^{mu'}  (share of accumulated error e' over Z_{mu'})
#[derive(Clone, Debug)]
pub struct PartialDecOutput {
    pub w_prime: [u128; 256],
    pub e_prime: [u128; 256],
}

/// Modulus switch one coefficient from Z_from to Z_to: round(to/from * x) mod to.
/// Identity when from == to (which holds in the active branch with q' = q).
fn mod_switch_coeff(x: u128, from: u128, to: u128) -> u128 {
    if from == to {
        return x % to;
    }
    let num = to.saturating_mul(x).saturating_add(from / 2);
    (num / from) % to
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::additive_ring;
    use crate::dealer_spdz::DealerSpdz;
    use crate::params::*;

    /// Sanity: summing all parties' w_prime shares gives the same as a single-
    /// party "v - <u,s>" mod q computation.
    #[test]
    fn test_partial_decrypt_w_prime_reconstructs() {
        type P = MlKem512;
        const K: usize = <P as MlKemParams>::K;
        let n = 3;
        let dealer = DealerSpdz::new(n, 20, 40, 2);
        let ks = dealer.generate_keypair::<P>();
        let dbl = dealer.generate_double_sharing(1);
        let s_full = additive_ring::reconstruct_vector_lifted(&ks.sk_shares, &dealer.thr);

        let q = dealer.thr.q as u32;
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

        let parties: Vec<PartySpdz<K>> = ks
            .sk_shares
            .iter()
            .zip(dbl.iter())
            .map(|(sk, d)| PartySpdz::new(sk.x, sk.clone(), d.clone(), dealer.thr))
            .collect();
        let min_id = parties.iter().map(|p| p.id).min().unwrap();

        let outs: Vec<_> = parties
            .iter()
            .map(|p| p.partial_decrypt(&u, &v, p.id == min_id))
            .collect();

        // Reference: <u, s_full> by direct negacyclic poly multiply.
        let mut reference = [0i128; 256];
        for j in 0..K {
            let s_j = s_full.data[j].data;
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
        let q_i = dealer.thr.q as i128;
        let q_u = dealer.thr.q;
        let mut expected = [0u128; 256];
        for c in 0..256 {
            let inner_mod = reference[c].rem_euclid(q_i) as u128;
            let v_term = v.data[c] as u128;
            expected[c] = if v_term >= inner_mod { v_term - inner_mod } else { q_u + v_term - inner_mod };
        }

        // Sum of parties' w_prime mod q' must equal expected (q' = q here).
        let q_prime = dealer.thr.q_prime;
        for c in 0..256 {
            let s = outs.iter().fold(0u128, |acc, o| (acc + o.w_prime[c]) % q_prime);
            assert_eq!(s, expected[c], "w_prime reconstruction mismatch at coeff {c}");
        }
    }

    /// Mask-and-open consistency over ell parallel sharings.
    #[test]
    fn test_mask_open_consistency_ell() {
        type P = MlKem512;
        const K: usize = <P as MlKemParams>::K;
        let ell = 4;
        let dealer = DealerSpdz::new(3, 20, 40, 2);
        let ks = dealer.generate_keypair::<P>();
        let dbl = dealer.generate_double_sharing(ell);

        use rand::Rng;
        let mut rng = rand::thread_rng();
        let parties: Vec<PartySpdz<K>> = ks
            .sk_shares
            .iter()
            .zip(dbl.iter())
            .map(|(sk, d)| PartySpdz::new(sk.x, sk.clone(), d.clone(), dealer.thr))
            .collect();

        let e_shares: Vec<[u128; 256]> = (0..parties.len())
            .map(|_| {
                let mut arr = [0u128; 256];
                for c in 0..256 {
                    arr[c] = rng.gen_range(0..dealer.thr.mu_prime);
                }
                arr
            })
            .collect();

        // Each party returns ell masked shares.
        let masked: Vec<Vec<[u128; 256]>> = parties
            .iter()
            .zip(e_shares.iter())
            .map(|(p, e)| p.mask(e))
            .collect();
        for m in &masked {
            assert_eq!(m.len(), ell);
        }

        let mu_prime = dealer.thr.mu_prime;
        for j in 0..ell {
            for c in 0..256 {
                let lhs = masked.iter().fold(0u128, |acc, m| (acc + m[j][c]) % mu_prime);
                let sum_e = e_shares.iter().fold(0u128, |acc, e| (acc + e[c]) % mu_prime);
                let sum_d = dbl.iter().fold(0u128, |acc, d| (acc + d.mu_prime[j][c]) % mu_prime);
                assert_eq!(lhs, (sum_e + sum_d) % mu_prime);
            }
        }
    }
}