//! Block 6 — Receiver + protocol driver for threshold decryption.
//!
//! Pipeline:
//!   offline:  dealer deals n Shamir key shares (any t reconstruct sk),
//!             plus t additive double sharings (one per active committee
//!             member; all t must be present at protocol time).
//!   step 1-3: each active member -> (<w'>^{q'}, <e'>^{mu'})
//!   step 4:   sum masked <e'>^{mu'} + <d_j>^{mu'} -> open e_tilde_j (mod mu')
//!   step 5:   each active member -> ell shares of <mu'*m>^{q'}
//!   step 6:   receiver sums per-j, decodes, majority-votes across ell -> 32 bytes
//!
//! Steps 1-5 are executed by the KBS committee (`threshold_decrypt`). Step 6
//! is the "local" portion executed inside the TEE (`receiver_reconstruct`).
//!
//! Active-committee convention:
//!   The active committee is the FIRST t parties (ids 1..=t), matching the
//!   `t` additive double-shares produced by the dealer. `assemble_parties`
//!   pairs the first t Shamir key shares with the t additive double shares.

use crate::dealer::{DoubleShare, ThrParams};
use crate::party::Party;
use crate::shamir_poly_ring::VectorShare;

/// Reconstruct the ell public e_tilde_j = e' + d_j (mod mu') by summing the
/// active members' masked shares for each j independently.
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
pub fn receiver_reconstruct(mu_m_shares: &[Vec<[u64; 256]>], thr: &ThrParams) -> [u8; 32] {
    assert!(!mu_m_shares.is_empty(), "receiver_reconstruct: no parties");
    let ell = mu_m_shares[0].len();
    for s in mu_m_shares.iter().skip(1) {
        assert_eq!(s.len(), ell, "receiver_reconstruct: parties disagree on ell");
    }

    let q_prime = thr.q_prime;
    let mu_prime = thr.mu_prime;
    let p = thr.p;

    let mut candidates = vec![[0u64; 256]; ell];
    for share_vec in mu_m_shares {
        for j in 0..ell {
            for c in 0..256 {
                candidates[j][c] = (candidates[j][c] + share_vec[j][c]) % q_prime;
            }
        }
    }

    assert_eq!(p, 2, "majority decoding currently implemented only for p = 2");

    let mut out = [0u8; 32];
    for c in 0..256 {
        let mut votes_for_one: u32 = 0;
        for j in 0..ell {
            let bit = decode_coeff(candidates[j][c], q_prime, mu_prime, p);
            votes_for_one += bit as u32;
        }
        let one_wins = (votes_for_one as usize) * 2 > ell;
        if one_wins {
            out[c / 8] |= 1u8 << (c % 8);
        }
    }
    out
}

fn decode_coeff(c: u64, q_prime: u64, mu_prime: u64, p: u64) -> u64 {
    let centered: i64 = {
        let r = (c % q_prime) as i64;
        if r >= (q_prime / 2) as i64 { r - q_prime as i64 } else { r }
    };
    let half = (mu_prime / 2) as i64;
    let rounded = if centered >= 0 {
        (centered + half) / mu_prime as i64
    } else {
        -((-centered + half) / mu_prime as i64)
    };
    (((rounded % p as i64) + p as i64) % p as i64) as u64
}

/// Distributed protocol (Steps 1-5) executed by the KBS committee.
///
/// `parties` MUST be the t active-committee members (ids 1..=t) — additive
/// double sharings only reconstruct when summed over the full active set.
pub fn threshold_decrypt<const K: usize, const D_U: usize, const D_V: usize>(
    parties: &[Party<K>],
    c: crate::kpke::Cyphertext<K, D_U, D_V>,
    thr: &ThrParams,
) -> Vec<Vec<[u64; 256]>> {
    assert!(!parties.is_empty(), "need >=1 party");
    let ell = parties[0].dbl.ell();
    for p in parties.iter().skip(1) {
        assert_eq!(p.dbl.ell(), ell, "parties disagree on ell");
    }

    let active_ids: Vec<u32> = parties.iter().map(|p| p.id).collect();
    let v_holder_id = *active_ids.iter().min().expect("need >=1 party");

    let step1: Vec<_> = parties
        .iter()
        .map(|p| {
            let is_holder = p.id == v_holder_id;
            p.partial_decrypt_compressed::<D_U, D_V>(c.clone(), &active_ids, is_holder)
        })
        .collect();

    let masked_per_party: Vec<Vec<[u64; 256]>> = parties
        .iter()
        .zip(step1.iter())
        .map(|(p, out)| p.mask(&out.e_prime))
        .collect();
    let e_tilde_opened = open_v(&masked_per_party, thr);

    parties
        .iter()
        .zip(step1.iter())
        .map(|(p, out)| {
            let is_holder = p.id == v_holder_id;
            p.finalize(&out.w_prime, &e_tilde_opened, is_holder)
        })
        .collect()
}

/// Assemble the t active-committee parties from dealer outputs.
///
/// Pairs the first t Shamir key shares (ids 1..=t) with the t additive
/// double shares produced by `generate_double_sharing`. The active committee
/// is fixed by convention to the FIRST t party ids.
pub fn assemble_parties<const K: usize>(
    sk_shares: &[VectorShare<K>],
    double_shares: &[DoubleShare],
    thr: ThrParams,
) -> Vec<Party<K>> {
    let t = double_shares.len();
    assert!(
        sk_shares.len() >= t,
        "need at least t = {t} Shamir key shares, got {}",
        sk_shares.len()
    );
    (0..t)
        .map(|i| {
            let sk = sk_shares[i].clone();
            let dbl = double_shares[i].clone();
            debug_assert_eq!(sk.x, dbl.party_id, "party id mismatch at slot {i}");
            Party::new(sk.x, sk, dbl, thr)
        })
        .collect()
}