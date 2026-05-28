//! Block 6 — Receiver + protocol driver for threshold decryption (Ajax Fig. 10).
//!
//! Ties together Dealer (Block 3), Party (Block 5), and the K-PKE encrypt path
//! to run the full mask-then-open protocol end to end:
//!
//!   offline:  dealer deals key shares + one double sharing
//!   step 1:   each active party -> (<u'>^eta, <e'>^Phi)
//!   step 2:   sum masked <e'>^Phi + <r>^Phi  ->  open v = e' + r  (mod Phi)
//!   step 3:   each active party -> <Phi*m>^eta
//!   step 4:   receiver sums <Phi*m>^eta (mod eta) -> Phi*m -> bits -> 32 bytes
//!
//! Representation reminder:
//!   * key shares: Shamir, NTT-form Ring (converted to additive via Lagrange
//!     inside Party::partial_decrypt).
//!   * double shares & all step-2/3 values: additive [u64;256] arrays over
//!     eta / Phi.

use crate::dealer::{DoubleShare, ThrParams};
use crate::party::Party;
use crate::shamir_ring::VectorShare;

/// Reconstruct the public v = e' + r (mod Phi) by summing the parties' masked
/// shares <v>^Phi := <e'>^Phi + <r>^Phi. Additive reconstruction = plain sum.
pub fn open_v(masked_shares: &[[u64; 256]], thr: &ThrParams) -> [u64; 256] {
    let phi = thr.phi;
    let mut v = [0u64; 256];
    for share in masked_shares {
        for c in 0..256 {
            v[c] = (v[c] + share[c]) % phi;
        }
    }
    v
}

/// Step 4 (receiver): reconstruct Phi*m from the additive shares <Phi*m>^eta
/// (sum mod eta), decode each coefficient to a bit, pack to 32 bytes
/// (LSB-first per byte, matching ML-KEM message convention).
pub fn receiver_reconstruct(phi_m_shares: &[[u64; 256]], thr: &ThrParams) -> [u8; 32] {
    let eta = thr.eta;
    let phi = thr.phi;
    let p = thr.p;

    // Sum shares mod eta -> Phi*m (+ residual noise) per coefficient.
    let mut phi_m = [0u64; 256];
    for share in phi_m_shares {
        for c in 0..256 {
            phi_m[c] = (phi_m[c] + share[c]) % eta;
        }
    }

    // Decode each coefficient to a bit: center mod eta, then round to nearest
    // multiple of Phi, reduce mod p.
    let mut out = [0u8; 32];
    for c in 0..256 {
        let bit = decode_coeff(phi_m[c], eta, phi, p);
        if bit != 0 {
            out[c / 8] |= 1u8 << (c % 8);
        }
    }
    out
}

/// Decode one Phi*m coefficient to a plaintext symbol in [0, p).
/// Mirrors the Python: c_centered = centered_mod(c, eta); round(c_centered/Phi) % p.
fn decode_coeff(c: u64, eta: u64, phi: u64, p: u64) -> u64 {
    // centered representative in (-eta/2, eta/2]
    let centered: i64 = {
        let r = (c % eta) as i64;
        if r >= (eta / 2) as i64 {
            r - eta as i64
        } else {
            r
        }
    };
    // round(centered / phi): add/sub half before integer division
    let half = (phi / 2) as i64;
    let rounded = if centered >= 0 {
        (centered + half) / phi as i64
    } else {
        -((-centered + half) / phi as i64)
    };
    // reduce mod p (p is power of two, small)
    (((rounded % p as i64) + p as i64) % p as i64) as u64
}

/// Convenience driver: run the full online protocol given the active committee
/// and a (cloneable) compressed ciphertext.
///
/// * `parties`: the active t+1 parties (each holds its key share + double share).
/// * `c`: the compressed MLWE ciphertext.
/// The smallest party id is designated the v-holder (mirrors Python id==0).
pub fn threshold_decrypt<const K: usize, const D_U: usize, const D_V: usize>(
    parties: &[Party<K>],
    c: crate::kpke::Cyphertext<K, D_U, D_V>,
    thr: &ThrParams,
) -> [u8; 32] {
    let active_ids: Vec<u32> = parties.iter().map(|p| p.id).collect();
    let v_holder_id = *active_ids.iter().min().expect("need >=1 party");

    // ----- Step 1: each party's (<u'>^eta, <e'>^Phi) ------------------------
    let step1: Vec<_> = parties
        .iter()
        .map(|p| {
            let is_holder = p.id == v_holder_id;
            p.partial_decrypt_compressed::<D_U, D_V>(c.clone(), &active_ids, is_holder)
        })
        .collect();

    // ----- Step 2: mask and open v = e' + r (mod Phi) -----------------------
    let masked: Vec<[u64; 256]> = parties
        .iter()
        .zip(step1.iter())
        .map(|(p, out)| p.mask(&out.e_prime_phi))
        .collect();
    let v_opened = open_v(&masked, thr);

    // ----- Step 3: each party's <Phi*m>^eta ---------------------------------
    let phi_m_shares: Vec<[u64; 256]> = parties
        .iter()
        .zip(step1.iter())
        .map(|(p, out)| {
            let is_holder = p.id == v_holder_id;
            p.finalize(&out.u_prime_eta, &v_opened, is_holder)
        })
        .collect();

    // ----- Step 4: receiver reconstructs the message ------------------------
    receiver_reconstruct(&phi_m_shares, thr)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dealer::Dealer;
    use crate::params::*;

    #[test]
    fn test_decode_coeff_bits() {
        let q = Q64;
        let eta = q - 1;
        let phi = eta / 2;
        // b=0 -> small values decode 0; b=1 -> near Phi decode 1.
        assert_eq!(decode_coeff(0, eta, phi, 2), 0);
        assert_eq!(decode_coeff(5, eta, phi, 2), 0);
        assert_eq!(decode_coeff(phi, eta, phi, 2), 1);
        assert_eq!(decode_coeff(phi + 7, eta, phi, 2), 1);
        assert_eq!(decode_coeff(phi - 7, eta, phi, 2), 1);
    }

    // NOTE: a full end-to-end test (deal -> encrypt -> threshold_decrypt ==
    // original message) requires wiring kpke::encrypt with a random message and
    // is best run as an integration test once lib.rs exposes the modules. The
    // unit tests here cover the receiver decode; per-block round-trips are
    // covered in dealer.rs / shamir_ring.rs / party.rs.
}