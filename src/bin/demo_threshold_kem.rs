//! Ladon active threshold KEM demo (SPDZ2k variant).
//!
//! Combines the threshold PKE protocol (Steps 1-5 + receiver_reconstruct)
//! with the Fujisaki-Okamoto finalization (SHA3 G/H/J), so the output is the
//! 32-byte shared key — what a KMS deployment actually returns.
//!
//! Pipeline:
//!   1. DealerSpdz deals lifted secret-key shares + ell parallel double sharings.
//!   2. Asset-owner side runs encaps under the public ek (FO).
//!   3. Committee runs threshold_decrypt (Steps 1-5).
//!   4. TEE side runs receiver_reconstruct (Step 6, majority over ell) and
//!      finalizes the FO transform (re-encrypt + compare; implicit rejection
//!      via crypt::j on failure).
//!
//! Convention: t = n - 1 (dishonest majority; ALL n parties participate).
//!
//! Build & run:
//!     cargo run --release --bin demo_threshold_kem_active

#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use bitvec::view::BitView;

use Ladon::additive_ring::reconstruct_vector_lifted;
use Ladon::crypt;
use Ladon::dealer_spdz::DealerSpdz;
use Ladon::kpke;
use Ladon::negacyclic::{decompress_ring_2k, decompress_vector_2k};
use Ladon::params::*;
use Ladon::party_spdz::PartySpdz;
use Ladon::ring::{Compressed, Ring};
use Ladon::serialize::{BitOrder, MlKemDeserialize, MlKemSerialize};
use Ladon::threshold_decrypt::{assemble_parties, receiver_reconstruct, threshold_decrypt};

// ===========================================================================
// ====== DEMO PARAMETERS — edit these =======================================
// ===========================================================================
const N_PARTIES: usize = 5;        // dishonest-majority: ALL must participate
const S_BITS:    u32   = 40;       // SPDZ2k statistical security parameter
const P_PLAINTEXT: u128 = 2;       // plaintext modulus
// Per-paramset ell (mirroring passive branch):
const ELL_LADON128: usize = 5;
const ELL_LADON256: usize = 32;
// ===========================================================================

fn main() {
    println!("==================================================================");
    println!("  Ladon Active Threshold KEM Demo (SPDZ2k)");
    println!("==================================================================");
    println!("  parties n         : {N_PARTIES}  (t = n-1 = {}, all must participate)", N_PARTIES - 1);
    println!("  k_bits / s_bits   : {K_BITS} / {S_BITS}");
    println!("  p (plaintext)     : {P_PLAINTEXT}");
    println!();
    run::<Ladon128>("Ladon128", ELL_LADON128);
    println!();
    run::<Ladon256>("Ladon256", ELL_LADON256);
    println!("==================================================================");
}

fn run<PARAMS: MlKemParams>(label: &str, ell: usize)
where
    [(); 960 * PARAMS::K + 32]:,
    [(); 1920 * PARAMS::K + 96]:,
    [(); PARAMS::K]:,
    [(); PARAMS::ETA_1]:,
    [(); PARAMS::ETA_2]:,
    [(); 64 * PARAMS::ETA_1]:,
    [(); 64 * PARAMS::ETA_2]:,
    [(); 32 * (PARAMS::D_U * PARAMS::K + PARAMS::D_V)]:,
    [(); 960 * PARAMS::K + 32]:,
{
    println!("------------------------------------------------------------------");
    println!("  {label}  (K = {}, ell = {ell})", PARAMS::K);
    println!("------------------------------------------------------------------");

    // ----- Offline: dealer ------------------------------------------------
    let dealer = DealerSpdz::new(N_PARTIES, K_BITS, S_BITS, P_PLAINTEXT);
    println!("  q  = 2^k                  = {}", dealer.thr.q);
    println!("  q' = 2^round(log2(q))     = {}", dealer.thr.q_prime);
    println!("  mu  = q  / p              = {}", dealer.thr.mu);
    println!("  mu' = q' / p              = {}", dealer.thr.mu_prime);

    let ks = dealer.generate_keypair::<PARAMS>();
    let dbl = dealer.generate_double_sharing(ell);
    println!("\n[Dealer] {} secret-key shares + {ell} double sharings dealt.", ks.sk_shares.len());

    // FO setup (asset-owner side): z (implicit-rejection seed), hash of ek.
    let z = crypt::random_bytes::<32>();
    let ek_hash = crypt::h(&ks.ek.serialize().into_vec());

    // ----- Encaps (asset owner) -------------------------------------------
    let m_bytes = crypt::random_bytes::<32>();
    let mut combined = [0u8; 64];
    combined[..32].copy_from_slice(&m_bytes);
    combined[32..].copy_from_slice(&ek_hash);
    let (key_a, rand) = crypt::g::<64>(&combined);

    let m: Compressed<1, Ring> =
        Compressed::<1, Ring>::deserialize(&m_bytes.view_bits::<BitOrder>().to_bitvec());
    let ct = kpke::encrypt_2k::<PARAMS>(ks.ek.clone(), m, rand);
    println!("[Encaps] key_a = {}", hex(&key_a));

    // ----- Threshold decapsulation ----------------------------------------
    let key_b = threshold_decaps::<PARAMS>(
        &ks.sk_shares,
        &dbl,
        &ks.ek,
        &z,
        &ek_hash,
        &ct,
        dealer.thr,
    );
    println!("[Decaps] key_b = {}", hex(&key_b));

    // ----- Verdict --------------------------------------------------------
    println!();
    if key_a == key_b {
        println!("  SUCCESS: threshold-decapsulated key matches encapsulated key.");
    } else {
        println!("  MISMATCH: keys differ.");
    }
}

/// Local "threshold_decaps" — runs the committee + TEE protocol and the FO
/// finalization to output the 32-byte shared key.
fn threshold_decaps<PARAMS: MlKemParams>(
    sk_shares: &[Ladon::additive_ring::VectorShare128<{ PARAMS::K }>],
    dbl: &[Ladon::dealer_spdz::DoubleShare],
    ek: &kpke::KpkeEncryptionKey<{ PARAMS::K }>,
    z: &[u8; 32],
    ek_hash: &[u8; 32],
    ct: &kpke::Cyphertext<{ PARAMS::K }, { PARAMS::D_U }, { PARAMS::D_V }>,
    thr: Ladon::additive_2k::SpdzParams,
) -> [u8; 32]
where
    [(); PARAMS::K]:,
    [(); PARAMS::ETA_1]:,
    [(); PARAMS::ETA_2]:,
    [(); 64 * PARAMS::ETA_1]:,
    [(); 64 * PARAMS::ETA_2]:,
    [(); 32 * (PARAMS::D_U * PARAMS::K + PARAMS::D_V)]:,
    [(); 960 * PARAMS::K + 32]:,
{
    // Decompress (u, v) for the parties.
    let u_dec =
        decompress_vector_2k::<{ PARAMS::K }>(&ct.0 .0, PARAMS::D_U as u32, K_BITS);
    let v_dec = decompress_ring_2k(&ct.1 .0, PARAMS::D_V as u32, K_BITS);

    // Committee: Steps 1-5 of the protocol.
    let parties: Vec<PartySpdz<{ PARAMS::K }>> =
        assemble_parties::<{ PARAMS::K }>(sk_shares, dbl, thr);
    let mu_m_shares = threshold_decrypt(&parties, &u_dec, &v_dec, &thr);

    // TEE: Step 6 (majority over ell) + FO finalization.
    let m_bytes = receiver_reconstruct(&mu_m_shares, &thr);
    let mut combined = [0u8; 64];
    combined[..32].copy_from_slice(&m_bytes);
    combined[32..].copy_from_slice(ek_hash);
    let (key_candidate, rand_prime) = crypt::g::<64>(&combined);

    // Re-encrypt to verify (FO check).
    let m: Compressed<1, Ring> =
        Compressed::<1, Ring>::deserialize(&m_bytes.view_bits::<BitOrder>().to_bitvec());
    let ct_prime = kpke::encrypt_2k::<PARAMS>(ek.clone(), m, rand_prime);
    let ct_match = ct_prime.0 == ct.0 && ct_prime.1 == ct.1;

    if ct_match {
        key_candidate
    } else {
        crypt::j([&z[..], ct.serialize().as_raw_slice()].concat())
    }
}

fn hex(b: &[u8]) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    for x in b {
        let _ = write!(s, "{:02x}", x);
    }
    s
}