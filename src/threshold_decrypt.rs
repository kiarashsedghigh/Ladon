//! Block 5 — SPDZ2k threshold-decrypt driver + receiver.
//!
//! Orchestrates Ajax Fig. 10 (skip-mod-switch variant) end to end:
//!
//!   offline: dealer deals lifted key shares + one (q, Delta) double sharing
//!   step 1 : each party -> (<u>^q, <e>^Delta)
//!   step 2 : sum (<e>^Delta + <r>^Delta) mod Delta  -> open v = e + r
//!   step 3 : each party -> <Delta*m>^q
//!   step 4 : receiver sums <Delta*m>^q mod q -> Delta*m -> bits -> 32 bytes
//!
//! Dishonest-majority: ALL n parties must participate; no committee selection,
//! no Lagrange. The smallest party id is the designated v-holder (mirrors the
//! Python `id==0` convention).

use crate::additive_2k::SpdzParams;
use crate::additive_ring::VectorShare128;
use crate::dealer_spdz::DoubleShare;
use crate::party_spdz::PartySpdz;
use crate::ring::{Ring, Vector};

/// Open v = e + r (mod Delta) by summing the parties' masked shares
/// <v>^Delta := <e>^Delta + <r>^Delta. Additive reconstruction = plain sum.
pub fn open_v(masked_shares: &[[u128; 256]], params: &SpdzParams) -> [u128; 256] {
    let delta = params.delta;
    let mut v = [0u128; 256];
    for share in masked_shares {
        for c in 0..256 {
            v[c] = (v[c] + share[c]) % delta;
        }
    }
    v
}

/// Step 4 (receiver): reconstruct Delta*m from the additive shares <Delta*m>^q
/// (sum mod q), decode each coefficient to a bit, pack to 32 bytes
/// (LSB-first per byte, matching ML-KEM message convention).
pub fn receiver_reconstruct(delta_m_shares: &[[u128; 256]], params: &SpdzParams) -> [u8; 32] {
    let q = params.q;
    let delta = params.delta;
    let p = params.p;

    // Sum shares mod q -> Delta*m (+ residual noise) per coefficient.
    let mut dm = [0u128; 256];
    for share in delta_m_shares {
        for c in 0..256 {
            dm[c] = (dm[c] + share[c]) % q;
        }
    }

    // Decode each coefficient to a bit: center mod q, round to nearest
    // multiple of Delta, reduce mod p.
    let mut out = [0u8; 32];
    for c in 0..256 {
        let bit = decode_coeff(dm[c], q, delta, p);
        if bit != 0 {
            out[c / 8] |= 1u8 << (c % 8);
        }
    }
    out
}

/// Decode one Delta*m coefficient to a plaintext symbol in [0, p).
/// Mirrors the Python: c_centered = centered_mod(c, q); round(c_centered/Delta) % p.
fn decode_coeff(c: u128, q: u128, delta: u128, p: u128) -> u128 {
    // centered representative in (-q/2, q/2]
    let r = c % q;
    let centered: i128 = if r >= q / 2 {
        r as i128 - q as i128
    } else {
        r as i128
    };
    let half = (delta / 2) as i128;
    let d = delta as i128;
    let rounded = if centered >= 0 {
        (centered + half) / d
    } else {
        -((-centered + half) / d)
    };
    let pi = p as i128;
    (((rounded % pi) + pi) % pi) as u128
}

/// Run the full SPDZ2k threshold-decrypt protocol given the n parties and a
/// DECOMPRESSED ciphertext (u, v).
///
/// All n parties must participate (dishonest majority, t = n-1). The smallest
/// party id is designated the v-holder.
pub fn threshold_decrypt<const K: usize>(
    parties: &[PartySpdz<K>],
    u: &Vector<K>,
    v: &Ring,
    params: &SpdzParams,
) -> [u8; 32] {
    let v_holder_id = parties.iter().map(|p| p.id).min().expect("need >=1 party");

    // ----- Step 1: each party's (<u>^q, <e>^Delta) -------------------------
    let step1: Vec<_> = parties
        .iter()
        .map(|p| p.partial_decrypt(u, v, p.id == v_holder_id))
        .collect();

    // ----- Step 2: mask and open v = e + r (mod Delta) ---------------------
    let masked: Vec<[u128; 256]> = parties
        .iter()
        .zip(step1.iter())
        .map(|(p, out)| p.mask(&out.e_share_delta))
        .collect();
    let v_opened = open_v(&masked, params);

    // ----- Step 3: each party's <Delta*m>^q ---------------------------------
    let delta_m_shares: Vec<[u128; 256]> = parties
        .iter()
        .zip(step1.iter())
        .map(|(p, out)| p.finalize(&out.u_share_q, &v_opened, p.id == v_holder_id))
        .collect();

    // ----- Step 4: receiver reconstructs the 32-byte message ---------------
    receiver_reconstruct(&delta_m_shares, params)
}

/// Helper to assemble PartySpdz objects from dealer outputs.
pub fn assemble_parties<const K: usize>(
    sk_shares: &[VectorShare128<K>],
    double_shares: &[DoubleShare],
    params: SpdzParams,
) -> Vec<PartySpdz<K>> {
    assert_eq!(sk_shares.len(), double_shares.len(), "share counts must match");
    sk_shares
        .iter()
        .zip(double_shares.iter())
        .map(|(sk, d)| PartySpdz::new(sk.x, sk.clone(), d.clone(), params))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::*;

    #[test]
    fn test_decode_coeff_bits() {
        let q: u128 = 1 << 20;
        let delta = q / 2;
        let p: u128 = 2;
        assert_eq!(decode_coeff(0, q, delta, p), 0);
        assert_eq!(decode_coeff(5, q, delta, p), 0);
        assert_eq!(decode_coeff(delta, q, delta, p), 1);
        assert_eq!(decode_coeff(delta + 7, q, delta, p), 1);
        assert_eq!(decode_coeff(delta - 7, q, delta, p), 1);
        assert_eq!(decode_coeff(q - 5, q, delta, p), 0);
    }

    // End-to-end is in the demo binary; here we just sanity-check the receiver
    // and the open_v helper with hand-crafted shares.
    #[test]
    fn test_open_v_sum() {
        let params = SpdzParams::new(20, 40, 2);
        let n = 3;
        // Three parties each hold a [u128; 256] masked share; the sum mod
        // Delta must be the open_v output.
        let mut shares = vec![[0u128; 256]; n];
        for i in 0..n {
            for c in 0..256 {
                shares[i][c] = ((i as u128 + 1) * (c as u128 + 1) * 7) % params.delta;
            }
        }
        let v = open_v(&shares, &params);
        for c in 0..256 {
            let expected = shares.iter().fold(0u128, |a, s| (a + s[c]) % params.delta);
            assert_eq!(v[c], expected);
        }
    }
}