//! Threshold decryption demo — set parameters below and run.
//!
//! Pipeline: Dealer deals (t,n) key shares + a double sharing; we K-PKE-encrypt
//! a random 32-byte message under the public key; a chosen committee of t+1
//! parties runs Ajax Fig. 10 threshold decryption; we check it equals both the
//! original message and the centralized kpke::decrypt output.
//!
//! Build: place this as src/bin/threshold_demo.rs (and declare the modules in
//! lib.rs: shamir, shamir_ring, dealer, party, threshold). Then:
//!     cargo run --bin threshold_demo
//!
//! Requires the crate's nightly features (generic_const_exprs) like the other
//! binaries in this project.

#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use bitvec::view::BitView;
use rand::{rngs::StdRng, RngCore, SeedableRng};

use Moiragus::dealer::Dealer;
use Moiragus::kpke;
use Moiragus::params::*;
use Moiragus::party::Party;
use Moiragus::ring::{Compressed, Ring};
use Moiragus::serialize::{BitOrder, MlKemDeserialize, MlKemSerialize};
use Moiragus::shamir_ring::reconstruct_vector;
use Moiragus::threshold::{assemble_parties, threshold_decrypt};

// ===========================================================================
// ====== DEMO PARAMETERS — edit these ======================================
// ===========================================================================
const N_PARTIES: usize = 7; // total parties the dealer shares to
const THRESHOLD: usize = 5; // t : any t+1 parties can decrypt
const P_PLAINTEXT: u64 = 2; // plaintext modulus (power of two); 2 = bit/coeff
// Which parties form the active committee (0-based indices, need t+1 of them):
const ACTIVE: &[usize] = &[0, 1, 2, 3, 4, 5, 6];
// Parameter set: MlKem512 (K=2), MlKem768 (K=3), or MlKem1024 (K=4).
type PARAMS = MlKem512;
// ===========================================================================

fn random_32() -> [u8; 32] {
    let mut rng = StdRng::from_entropy();
    let mut out = [0u8; 32];
    rng.fill_bytes(&mut out);
    out
}

fn hex(b: &[u8]) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    for x in b {
        let _ = write!(s, "{:02x}", x);
    }
    s
}

fn main() {
    // main() cannot carry generics or a where-clause, so all the work — and the
    // generic_const_exprs bounds — live in run::<PARAMS>().
    run::<PARAMS>();
}

fn run<PARAMS: MlKemParams>()
where
    [(); 384 * PARAMS::K + 32]:,
    [(); 768 * PARAMS::K + 96]:,
    [(); PARAMS::K]:,
    [(); PARAMS::ETA_1]:,
    [(); PARAMS::ETA_2]:,
    [(); 64 * PARAMS::ETA_1]:,
    [(); 64 * PARAMS::ETA_2]:,
    [(); 32 * (PARAMS::D_U * PARAMS::K + PARAMS::D_V)]:,
{
    assert!(
        ACTIVE.len() >= THRESHOLD + 1,
        "need at least t+1 = {} active parties, got {}",
        THRESHOLD + 1,
        ACTIVE.len()
    );

    println!("==================================================================");
    println!("  Threshold K-PKE Decryption Demo (Ajax Fig. 10, mask-then-open)");
    println!("==================================================================");
    println!("  parties n      : {N_PARTIES}");
    println!("  threshold t     : {THRESHOLD}  (any {} can decrypt)", THRESHOLD + 1);
    println!("  active committee : {ACTIVE:?}");
    println!("  K (module rank) : {}", PARAMS::K);
    println!("  q               : {}", Q);
    println!("  p               : {P_PLAINTEXT}");

    // ---- Offline: dealer deals key shares + a double sharing ---------------
    let dealer = Dealer::new(THRESHOLD, N_PARTIES, P_PLAINTEXT);
    println!(
        "  eta = q-1       : {}\n  Phi = eta/p     : {}\n  Delta = q/p     : {}",
        dealer.thr.eta, dealer.thr.phi, dealer.thr.delta
    );

    let ks = dealer.generate_keypair::<PARAMS>();
    let dbl = dealer.generate_double_sharing();
    println!("\n[Dealer] dealt {} secret-key shares + double sharing.", ks.sk_shares.len());

    // Sanity: any t+1 shares reconstruct the same (NTT-form) key.
    let rec = reconstruct_vector(&ks.sk_shares[0..THRESHOLD + 1].to_vec());
    let rec2 = reconstruct_vector(&ks.sk_shares[N_PARTIES - THRESHOLD - 1..N_PARTIES].to_vec());
    assert_eq!(rec, rec2, "key shares inconsistent");
    println!("[Check ] two quorums reconstruct identical secret key. OK");

    // ---- Encrypt a random message under the public key ---------------------
    let msg = random_32();
    let m: Compressed<1, Ring> =
        Compressed::<1, Ring>::deserialize(&msg.view_bits::<BitOrder>().to_bitvec());
    let rand = random_32();
    let ct = kpke::encrypt::<PARAMS>(ks.ek.clone(), m, rand);
    println!("\n[Enc   ] message  : {}", hex(&msg));

    // ---- Centralized decrypt (reference) ----------------------------------
    // Reconstruct the full key once, just to compare against the threshold run.
    let full_sk = reconstruct_vector(&ks.sk_shares[0..THRESHOLD + 1].to_vec());
    let m_central = kpke::decrypt::<PARAMS>(full_sk, ct.clone());
    let central_bytes: Vec<u8> = m_central.serialize().into_vec();
    println!("[Dec C ] central  : {}", hex(&central_bytes));

    // ---- Threshold decrypt with the active committee ----------------------
    let parties: Vec<Party<{ PARAMS::K }>> =
        assemble_parties::<{ PARAMS::K }>(&ks.sk_shares, &dbl, ACTIVE, dealer.thr);
    // D_U / D_V are inferred from ct's concrete Cyphertext type.
    let recovered = threshold_decrypt(&parties, ct, &dealer.thr);
    println!("[Dec T ] threshold: {}", hex(&recovered));

    // ---- Verdict ----------------------------------------------------------
    println!("\n------------------------------------------------------------------");
    let ok_central = recovered.as_slice() == central_bytes.as_slice();
    let ok_orig = recovered == msg;
    if ok_orig {
        println!("  SUCCESS: threshold output == original message.");
    } else if ok_central {
        println!("  PARTIAL: threshold == central decrypt, but both != original");
        println!("           (a K-PKE decryption failure, independent of threshold).");
    } else {
        println!("  MISMATCH: threshold output differs from central decrypt.");
        println!("            Check parameters / a decryption-failure event.");
    }
    println!("==================================================================");
}