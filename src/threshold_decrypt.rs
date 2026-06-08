//! Block 5 — SPDZ2k threshold-decrypt driver + receiver.
//!
//! Orchestrates the SPDZ2k threshold protocol end to end with the same
//! committee/TEE split as the passive branch:
//!
//!   offline:  dealer deals lifted key shares + ell parallel double sharings
//!   step 1-3: each party -> (<w'>^{q'}, <e'>^{mu'})
//!   step 4:   sum (<e'>^{mu'} + <d_j>^{mu'}) mod mu' -> open e_tilde_j
//!   step 5:   each party -> ell shares of <mu'*m>^{q'}
//!   step 6:   receiver sums per-j, decodes, majority-votes across ell
//!
//! Steps 1-5 are the "distributed" portion (`threshold_decrypt`).
//! Step 6 is the "local" TEE portion (`receiver_reconstruct`).
//!
//! Dishonest-majority: ALL n parties must participate; no committee
//! selection, no Lagrange. The smallest party id is the v-holder.

use crate::additive_2k::SpdzParams;
use crate::additive_ring::VectorShare128;
use crate::dealer_spdz::DoubleShare;
use crate::party_spdz::PartySpdz;
use crate::ring::{Ring, Vector};

/// Reconstruct the ell public e_tilde_j = e' + d_j (mod mu') by summing
/// each party's masked shares for each j independently.
/// `masked_shares[i]` is party i's Vec of length ell.
pub fn open_e_tilde(masked_shares: &[Vec<[u128; 256]>], thr: &SpdzParams) -> Vec<[u128; 256]> {
    assert!(!masked_shares.is_empty(), "open_e_tilde: no parties");
    let ell = masked_shares[0].len();
    for s in masked_shares.iter().skip(1) {
        assert_eq!(s.len(), ell, "open_e_tilde: parties disagree on ell");
    }
    let mu_prime = thr.mu_prime;

    let mut e_tilde_per_round = vec![[0u128; 256]; ell];
    for share_vec in masked_shares {
        for j in 0..ell {
            for c in 0..256 {
                e_tilde_per_round[j][c] =
                    (e_tilde_per_round[j][c] + share_vec[j][c]) % mu_prime;
            }
        }
    }
    e_tilde_per_round
}

/// Step 6 (TEE-local) with ell-fold majority decoding.
///
/// `mu_m_shares[i]` is party i's Vec of length ell, where entry j is its
/// additive share of (mu' * m)_j over Z_{q'}. The receiver:
///   1) sums shares mod q' per (j, coefficient) to get ell candidate mu'*m
///      polynomials,
///   2) decodes each candidate coefficient to a bit,
///   3) takes coefficient-wise majority across the ell decoded bits,
///   4) packs 256 majority-voted bits into 32 bytes.
///
/// Section 5.1 of the paper: a single double sharing has per-coeff failure
/// probability ~ |e'|/mu'; majority over ell drops the per-coeff failure
/// probability like a Binomial(ell, p) tail.
pub fn receiver_reconstruct(mu_m_shares: &[Vec<[u128; 256]>], thr: &SpdzParams) -> [u8; 32] {
    assert!(!mu_m_shares.is_empty(), "receiver_reconstruct: no parties");
    let ell = mu_m_shares[0].len();
    for s in mu_m_shares.iter().skip(1) {
        assert_eq!(s.len(), ell, "receiver_reconstruct: parties disagree on ell");
    }

    let q_prime = thr.q_prime;
    let mu_prime = thr.mu_prime;
    let p = thr.p;

    // 1) Sum shares mod q' per (j, c) -> ell candidate mu'*m vectors.
    let mut candidates = vec![[0u128; 256]; ell];
    for share_vec in mu_m_shares {
        for j in 0..ell {
            for c in 0..256 {
                candidates[j][c] = (candidates[j][c] + share_vec[j][c]) % q_prime;
            }
        }
    }

    // 2)+3) Decode ell candidate bits per coefficient and majority-vote.
    assert_eq!(p, 2, "majority decoding currently implemented only for p = 2");

    let mut out = [0u8; 32];
    for c in 0..256 {
        let mut votes_for_one: u32 = 0;
        for j in 0..ell {
            let bit = decode_coeff(candidates[j][c], q_prime, mu_prime, p);
            votes_for_one += bit as u32;
        }
        // Strict-majority rule for ones: votes_for_one > ell/2.
        // Ties (even ell only) default to 0.
        let one_wins = (votes_for_one as usize) * 2 > ell;
        if one_wins {
            out[c / 8] |= 1u8 << (c % 8);
        }
    }
    out
}

/// Decode one (mu' * m) coefficient to a plaintext symbol in [0, p).
fn decode_coeff(c: u128, q_prime: u128, mu_prime: u128, p: u128) -> u128 {
    let r = c % q_prime;
    let centered: i128 = if r >= q_prime / 2 {
        r as i128 - q_prime as i128
    } else {
        r as i128
    };
    let half = (mu_prime / 2) as i128;
    let m = mu_prime as i128;
    let rounded = if centered >= 0 {
        (centered + half) / m
    } else {
        -((-centered + half) / m)
    };
    let pi = p as i128;
    (((rounded % pi) + pi) % pi) as u128
}

/// Distributed protocol (Steps 1-5) executed by the n-party committee.
/// Returns the per-party shares of (mu' * m)_j over Z_{q'}, indexed as
/// `out[i][j]` for the i-th active party and the j-th of the ell parallel
/// double sharings.
///
/// Dishonest-majority: ALL parties in `parties` must participate.
pub fn threshold_decrypt<const K: usize>(
    parties: &[PartySpdz<K>],
    u: &Vector<K>,
    v: &Ring,
    thr: &SpdzParams,
) -> Vec<Vec<[u128; 256]>> {
    assert!(!parties.is_empty(), "need >=1 party");
    let ell = parties[0].dbl.ell();
    for p in parties.iter().skip(1) {
        assert_eq!(p.dbl.ell(), ell, "parties disagree on ell");
    }

    let v_holder_id = parties.iter().map(|p| p.id).min().expect("need >=1 party");

    // ----- Steps 1-3: each party's (<w'>^{q'}, <e'>^{mu'}) -----------------
    let step1: Vec<_> = parties
        .iter()
        .map(|p| p.partial_decrypt(u, v, p.id == v_holder_id))
        .collect();

    // ----- Step 4: mask and open ell parallel e_tilde_j (mod mu') ----------
    let masked_per_party: Vec<Vec<[u128; 256]>> = parties
        .iter()
        .zip(step1.iter())
        .map(|(p, out)| p.mask(&out.e_prime))
        .collect();
    let e_tilde_opened = open_e_tilde(&masked_per_party, thr);

    // ----- Step 5: each party's ell shares of <mu'*m>_j over Z_{q'} --------
    parties
        .iter()
        .zip(step1.iter())
        .map(|(p, out)| p.finalize(&out.w_prime, &e_tilde_opened, p.id == v_holder_id))
        .collect()
}

/// Helper to assemble PartySpdz objects from dealer outputs.
pub fn assemble_parties<const K: usize>(
    sk_shares: &[VectorShare128<K>],
    double_shares: &[DoubleShare],
    thr: SpdzParams,
) -> Vec<PartySpdz<K>> {
    assert_eq!(sk_shares.len(), double_shares.len(), "share counts must match");
    sk_shares
        .iter()
        .zip(double_shares.iter())
        .map(|(sk, d)| PartySpdz::new(sk.x, sk.clone(), d.clone(), thr))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::*;

    #[test]
    fn test_decode_coeff_bits() {
        let q_prime: u128 = 1 << 20;
        let mu_prime = q_prime / 2;
        let p: u128 = 2;
        assert_eq!(decode_coeff(0, q_prime, mu_prime, p), 0);
        assert_eq!(decode_coeff(mu_prime, q_prime, mu_prime, p), 1);
        assert_eq!(decode_coeff(mu_prime + 7, q_prime, mu_prime, p), 1);
        assert_eq!(decode_coeff(q_prime - 5, q_prime, mu_prime, p), 0);
    }

    #[test]
    fn test_open_e_tilde_sum() {
        let thr = SpdzParams::new(20, 40, 2);
        let n = 3;
        let ell = 4;
        let mut shares: Vec<Vec<[u128; 256]>> = vec![vec![[0u128; 256]; ell]; n];
        for i in 0..n {
            for j in 0..ell {
                for c in 0..256 {
                    shares[i][j][c] = ((i as u128 + 1) * (j as u128 + 1) * (c as u128 + 1) * 7)
                        % thr.mu_prime;
                }
            }
        }
        let e_tilde = open_e_tilde(&shares, &thr);
        assert_eq!(e_tilde.len(), ell);
        for j in 0..ell {
            for c in 0..256 {
                let expected = shares.iter().fold(0u128, |a, s| (a + s[j][c]) % thr.mu_prime);
                assert_eq!(e_tilde[j][c], expected);
            }
        }
    }
}