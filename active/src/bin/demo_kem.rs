//! Centralized Ladon KEM round-trip (active / SPDZ2k variant).
//!
//! No threshold, no parties, no sharing — just the centralized PKE-over-2^k
//! wrapped with the Fujisaki-Okamoto transform (SHA3 G/H/J). Mirrors the
//! passive branch's `demo_kem.rs` but uses `kpke::*_2k` so it works in the
//! q = 2^k regime.
//!
//! Build & run:
//!     cargo run --release --bin demo_kem_active

#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use bitvec::view::BitView;

use Ladon::crypt;
use Ladon::kpke;
use Ladon::params::*;
use Ladon::ring::{Compressed, Ring};
use Ladon::serialize::{BitOrder, MlKemDeserialize, MlKemSerialize};

fn main() {
    println!("==================================================================");
    println!("  Ladon Centralized KEM Round-Trip (active / SPDZ2k variant)");
    println!("==================================================================");
    println!();
    run::<Ladon128>("Ladon128");
    println!();
    run::<Ladon256>("Ladon256");
    println!("==================================================================");
}

fn run<PARAMS: MlKemParams>(label: &str)
where
    [(); 960 * PARAMS::K + 32]:,
    [(); 1920 * PARAMS::K + 96]:,
    [(); PARAMS::K]:,
    [(); PARAMS::ETA_1]:,
    [(); PARAMS::ETA_2]:,
    [(); 64 * PARAMS::ETA_1]:,
    [(); 64 * PARAMS::ETA_2]:,
    [(); 32 * (PARAMS::D_U * PARAMS::K + PARAMS::D_V)]:,
{
    println!("------------------------------------------------------------------");
    println!("  {label}  (K = {}, q = 2^{K_BITS})", PARAMS::K);
    println!("------------------------------------------------------------------");

    // ----- KeyGen + FO setup -----------------------------------------------
    let (ek, sk) = kpke::key_gen_2k::<PARAMS>();
    let z = crypt::random_bytes::<32>();
    let ek_hash = crypt::h(&ek.serialize().into_vec());

    // ----- Encaps ----------------------------------------------------------
    let m_bytes = crypt::random_bytes::<32>();
    let mut combined = [0u8; 64];
    combined[..32].copy_from_slice(&m_bytes);
    combined[32..].copy_from_slice(&ek_hash);
    let (key_a, rand) = crypt::g::<64>(&combined);

    let m: Compressed<1, Ring> =
        Compressed::<1, Ring>::deserialize(&m_bytes.view_bits::<BitOrder>().to_bitvec());
    let ct = kpke::encrypt_2k::<PARAMS>(ek.clone(), m, rand);

    // ----- Decaps (FO finalization) ---------------------------------------
    let m_recovered = kpke::decrypt_2k::<PARAMS>(sk, ct.clone());
    let m_recovered_bytes: Vec<u8> = m_recovered.serialize().into_vec();

    let mut combined2 = [0u8; 64];
    combined2[..32].copy_from_slice(&m_recovered_bytes[..32]);
    combined2[32..].copy_from_slice(&ek_hash);
    let (key_b_candidate, rand_prime) = crypt::g::<64>(&combined2);

    let ct_prime =
        kpke::encrypt_2k::<PARAMS>(ek.clone(), m_recovered.clone(), rand_prime);
    let ct_match = ct_prime.0 == ct.0 && ct_prime.1 == ct.1;

    let key_b = if ct_match {
        key_b_candidate
    } else {
        crypt::j([&z[..], ct.serialize().as_raw_slice()].concat())
    };

    println!("  message    : {}", hex(&m_bytes));
    println!("  key_a      : {}", hex(&key_a));
    println!("  key_b      : {}", hex(&key_b));
    println!("  ct match   : {}", if ct_match { "yes (FO check passed)" } else { "no (implicit rejection)" });
    println!();
    if key_a == key_b {
        println!("  SUCCESS: shared keys agree.");
    } else {
        println!("  MISMATCH: shared keys differ.");
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