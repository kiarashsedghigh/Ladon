//! SPDZ2k threshold-decryption demo — set parameters below and run.
//!
//! Pipeline: DealerSpdz deals lifted secret-key shares + one (q, Delta) double
//! sharing; we K-PKE-encrypt (under q=2^k via encrypt_2k) a random 32-byte
//! message under the public key; all n parties run the SPDZ2k threshold
//! decrypt; we check it equals both the original message and the centralized
//! decrypt_2k output.
//!
//! Build: place this as src/bin/threshold_spdz_demo.rs (and declare these
//! modules in lib.rs: additive_2k, additive_ring, dealer_spdz, party_spdz,
//! threshold_spdz, negacyclic). Then:
//!
//!     cargo run --bin threshold_spdz_demo
//!
//! Requires the crate's nightly features (generic_const_exprs) like the other
//! binaries in this project.

#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use bitvec::view::BitView;
use rand::{rngs::StdRng, RngCore, SeedableRng};

use Ladon::additive_ring::reconstruct_vector_lifted;
use Ladon::dealer_spdz::DealerSpdz;
use Ladon::kpke;
use Ladon::params::*;
use Ladon::party_spdz::PartySpdz;
use Ladon::ring::{Compressed, Ring};
use Ladon::serialize::{BitOrder, MlKemDeserialize, MlKemSerialize};
use Ladon::threshold_decrypt::{assemble_parties, threshold_decrypt};

// ===========================================================================
// ====== DEMO PARAMETERS — edit these =======================================
// ===========================================================================
const N_PARTIES: usize = 5;     // dishonest-majority: ALL n must participate
const K_BITS:    u32   = 30;    // base PKE modulus 2^k. MUST match kpke::K_BITS.
const S_BITS:    u32   = 40;    // statistical security (SPDZ2k lift width)
const P_PLAINTEXT: u128 = 2;    // plaintext modulus (power of two)
// Parameter set: MlKem512 / MlKem768 / MlKem1024.
type PARAMS = MlKem512;
// ===========================================================================

fn main() {
    run::<PARAMS>();
}

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
    println!("==================================================================");
    println!("  SPDZ2k Threshold K-PKE Decryption Demo");
    println!("==================================================================");
    println!("  parties n       : {N_PARTIES}  (ALL must participate)");
    println!("  base k          : {K_BITS}  -> q = 2^{K_BITS} = {}", 1u128 << K_BITS);
    println!("  stat. sec. s    : {S_BITS}  -> shares in 2^{}", K_BITS + S_BITS);
    println!("  p (plaintext)   : {P_PLAINTEXT}");
    println!("  K (module rank) : {}", PARAMS::K);

    // ---- Offline: dealer deals key shares + a (q, Delta) double sharing ----
    let dealer = DealerSpdz::new(N_PARTIES, K_BITS, S_BITS, P_PLAINTEXT);
    println!(
        "  Delta = q/p     : {}\n  m_share = 2^(k+s): {}",
        dealer.params.delta, dealer.params.m_share
    );

    let ks = dealer.generate_keypair::<PARAMS>();
    let dbl = dealer.generate_double_sharing();
    println!(
        "\n[Dealer] dealt {} lifted secret-key shares + double sharing.",
        ks.sk_shares.len()
    );

    // Sanity: reconstructing from all n shares is order-independent.
    let rec_a = reconstruct_vector_lifted(&ks.sk_shares, &dealer.params);
    let mut reversed = ks.sk_shares.clone();
    reversed.reverse();
    let rec_b = reconstruct_vector_lifted(&reversed, &dealer.params);
    assert_eq!(rec_a, rec_b, "key shares inconsistent");
    println!("[Check ] reconstruction is order-independent. OK");

    // ---- Encrypt a random message under the 2^k PKE -----------------------
    let msg = random_32();
    let m: Compressed<1, Ring> =
        Compressed::<1, Ring>::deserialize(&msg.view_bits::<BitOrder>().to_bitvec());
    let rand = random_32();
    let ct = kpke::encrypt_2k::<PARAMS>(ks.ek.clone(), m, rand);
    println!("\n[Enc   ] message  : {}", hex(&msg));

    // ---- Centralized decrypt (reference) ----------------------------------
    let m_central = kpke::decrypt_2k::<PARAMS>(rec_a.clone(), ct.clone());
    let central_bytes: Vec<u8> = m_central.serialize().into_vec();
    println!("[Dec C ] central  : {}", hex(&central_bytes));

    // ---- Threshold decrypt (all n parties) --------------------------------
    // partial_decrypt takes the DECOMPRESSED (u, v); decompress here using
    // the same shift-based helpers encrypt_2k/decrypt_2k use internally.
    let u_decompressed = Ladon::negacyclic::decompress_vector_2k::<{ PARAMS::K }>(
        &ct.0 .0,
        PARAMS::D_U as u32,
        K_BITS,
    );
    let v_decompressed =
        Ladon::negacyclic::decompress_ring_2k(&ct.1 .0, PARAMS::D_V as u32, K_BITS);

    let parties: Vec<PartySpdz<{ PARAMS::K }>> =
        assemble_parties::<{ PARAMS::K }>(&ks.sk_shares, &dbl, dealer.params);
    let recovered = threshold_decrypt(&parties, &u_decompressed, &v_decompressed, &dealer.params);
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
        println!("            Check (k, s, p) consistency with kpke::K_BITS.");
    }
    println!("==================================================================");
}