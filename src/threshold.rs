//! Block 6 — Receiver + protocol driver for threshold decryption.
//!
//! Notation matches the paper (q, q', mu, mu' ; w, w', e').
//!
//! Pipeline:
//!   offline:  dealer deals key shares + ell parallel double sharings
//!   step 1-3: each active party -> (<w'>^{q'}, <e'>^{mu'})
//!   step 4:   sum masked <e'>^{mu'} + <d_j>^{mu'} -> open e_tilde_j (mod mu')
//!   step 5:   each active party -> ell shares of <mu'*m>^{q'}
//!   step 6:   receiver sums per-j, decodes, majority-votes across ell -> 32 bytes
//!
//! Steps 1-5 are the "distributed" portion executed by the KBS committee
//! (`threshold_decrypt`). Step 6 is the "local" portion executed inside the
//! TEE (`receiver_reconstruct`). The caller invokes both in sequence.
//!
//! Representation reminder:
//!   * key shares: Shamir, NTT-form Ring (converted to additive via Lagrange
//!     inside Party::partial_decrypt).
//!   * double shares & all step-4/5 values: additive [u64;256] arrays over
//!     q' / mu'.

use crate::dealer::{DoubleShare, ThrParams};
use crate::party::Party;
use crate::shamir_poly_ring::VectorShare;

/// Reconstruct the ell public e_tilde_j = e' + d_j (mod mu') by summing the
/// parties' masked shares for each j independently. `masked_shares[i]` is
/// party i's Vec of length ell. Returns a Vec of length ell whose j-th entry
/// is the reconstructed e_tilde_j.
pub fn open_v(masked_shares: &[Vec<[u64; 256]>], thr: &ThrParams) -> Vec<[u64; 256]> {
    assert!(!masked_shares.is_empty(), "open_v: no parties");
    let ell = masked_shares[0].len();
    for s in masked_shares.iter().skip(1) {
        assert_eq!(s.len(), ell, "open_v: parties disagree on ell");
    }
    let mu_prime = thr.mu_prime;

    let mut e_tilde_per_round = vec![[0u64; 256]; ell];
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

/// Step 6 (receiver, local-to-TEE) with ell-fold majority decoding.
///
/// `mu_m_shares[i]` is party i's Vec of length ell, where entry j is its
/// additive share of (mu' * m)_j over Z_{q'}. The receiver:
///   1) sums shares mod q' per (j, coefficient) to get ell candidate mu'*m
///      polynomials,
///   2) decodes each candidate coefficient to a bit,
///   3) takes coefficient-wise majority across the ell decoded bits,
///   4) packs 256 majority-voted bits into 32 bytes (LSB-first per byte).
///
/// This is the amplification in Section 5.1 of the paper: a single double
/// sharing has per-coeff failure prob ~ |e'|/mu'; majority over ell drops
/// the per-coeff failure prob like Binomial(ell, p) tails (and the whole-
/// message success is the product over the N coefficients).
///
/// Called AFTER `threshold_decrypt` on its output.
pub fn receiver_reconstruct(mu_m_shares: &[Vec<[u64; 256]>], thr: &ThrParams) -> [u8; 32] {
    assert!(!mu_m_shares.is_empty(), "receiver_reconstruct: no parties");
    let ell = mu_m_shares[0].len();
    for s in mu_m_shares.iter().skip(1) {
        assert_eq!(s.len(), ell, "receiver_reconstruct: parties disagree on ell");
    }

    let q_prime = thr.q_prime;
    let mu_prime = thr.mu_prime;
    let p = thr.p;

    // 1) Sum shares mod q' per (j, c) -> ell candidate mu'*m vectors.
    let mut candidates = vec![[0u64; 256]; ell];
    for share_vec in mu_m_shares {
        for j in 0..ell {
            for c in 0..256 {
                candidates[j][c] = (candidates[j][c] + share_vec[j][c]) % q_prime;
            }
        }
    }

    // 2)+3) For each coefficient c, decode ell candidate bits and majority vote.
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

/// Decode one (mu'*m) coefficient to a plaintext symbol in [0, p).
/// Center mod q', then round to nearest multiple of mu', reduce mod p.
fn decode_coeff(c: u64, q_prime: u64, mu_prime: u64, p: u64) -> u64 {
    // centered representative in (-q'/2, q'/2]
    let centered: i64 = {
        let r = (c % q_prime) as i64;
        if r >= (q_prime / 2) as i64 {
            r - q_prime as i64
        } else {
            r
        }
    };
    // round(centered / mu'): add/sub half before integer division
    let half = (mu_prime / 2) as i64;
    let rounded = if centered >= 0 {
        (centered + half) / mu_prime as i64
    } else {
        -((-centered + half) / mu_prime as i64)
    };
    // reduce mod p (p is power of two, small)
    (((rounded % p as i64) + p as i64) % p as i64) as u64
}

/// Distributed protocol (Steps 1-5) executed by the KBS committee.
///
/// * `parties`: the active parties (each holds its key share + a length-ell
///   double-share batch).
/// * `c`: the compressed MLWE ciphertext.
/// The smallest party id is designated the v-holder (mirrors Python id==0).
///
/// Returns the per-party shares of (mu' * m)_j over Z_{q'}, indexed as
/// `out[i][j]` for the i-th active party and the j-th of the ell parallel
/// double sharings. To recover the 32-byte plaintext, pass this directly to
/// `receiver_reconstruct(&out, thr)`, which performs Step 6 locally on the
/// TEE side (sum, decode, coefficient-wise majority, pack).
///
/// Splitting at this seam mirrors the deployment-time architecture: the KBS
/// committee jointly produces the per-party (mu' * m)_j shares, and the TEE
/// then runs `receiver_reconstruct` locally inside the enclave.
pub fn threshold_decrypt<const K: usize, const D_U: usize, const D_V: usize>(
    parties: &[Party<K>],
    c: crate::kpke::Cyphertext<K, D_U, D_V>,
    thr: &ThrParams,
) -> Vec<Vec<[u64; 256]>> {
    assert!(!parties.is_empty(), "need >=1 party");
    // Consistency: every party must carry the same ell.
    let ell = parties[0].dbl.ell();
    for p in parties.iter().skip(1) {
        assert_eq!(p.dbl.ell(), ell, "parties disagree on ell");
    }

    let active_ids: Vec<u32> = parties.iter().map(|p| p.id).collect();
    let v_holder_id = *active_ids.iter().min().expect("need >=1 party");

    // ----- Steps 1-3: each party's (<w'>^{q'}, <e'>^{mu'}) -----------------
    let step1: Vec<_> = parties
        .iter()
        .map(|p| {
            let is_holder = p.id == v_holder_id;
            p.partial_decrypt_compressed::<D_U, D_V>(c.clone(), &active_ids, is_holder)
        })
        .collect();

    // ----- Step 4: mask and open ell parallel e_tilde_j (mod mu') -----------
    let masked_per_party: Vec<Vec<[u64; 256]>> = parties
        .iter()
        .zip(step1.iter())
        .map(|(p, out)| p.mask(&out.e_prime))
        .collect();
    let e_tilde_opened: Vec<[u64; 256]> = open_v(&masked_per_party, thr);

    // ----- Step 5: each party's ell shares of <mu'*m>_j over Z_{q'} --------
    parties
        .iter()
        .zip(step1.iter())
        .map(|(p, out)| {
            let is_holder = p.id == v_holder_id;
            p.finalize(&out.w_prime, &e_tilde_opened, is_holder)
        })
        .collect()
}

/// Helper to assemble Party objects from dealer outputs for a chosen active set.
/// `indices` selects which of the n dealt shares form the committee (0-based).
pub fn assemble_parties<const K: usize>(
    sk_shares: &[VectorShare<K>],
    double_shares: &[DoubleShare],
    indices: &[usize],
    thr: ThrParams,
) -> Vec<Party<K>> {
    indices
        .iter()
        .map(|&i| {
            let sk = sk_shares[i].clone();
            let dbl = double_shares[i].clone();
            Party::new(sk.x, sk, dbl, thr)
        })
        .collect()
}