//! Demo: Ladon centralized KEM round-trip (KeyGen → Encaps → Decaps).
//!
//! This demo runs the asset-owner / single-key-holder flow end-to-end:
//! the same secret key holder performs both encaps and decaps. It is the
//! baseline that the threshold demo (demo_threshold_kem) replaces with a
//! committee-coordinated decapsulation.
//!
//! Build & run:
//!     cargo run --release --bin demo_kem

#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use Ladon::mlkem::{self, MlKemCyphertext, MlKemDecapsulationKey, MlKemEncapsulationKey};
use Ladon::params::*;
use Ladon::serialize::{MlKemDeserialize, MlKemSerialize};

fn main() {
    println!("==================================================================");
    println!("  Ladon centralized KEM round-trip demo");
    println!("==================================================================");
    println!();
    println!("  What this demo does:");
    println!("    1. KeyGen  - generate an encapsulation key (ek, public) and a");
    println!("                 decapsulation key (dk, private).");
    println!("    2. Encaps  - sample a fresh 32-byte secret m, encrypt it under");
    println!("                 ek, and derive a shared key K_B = G(m, H(ek)).");
    println!("    3. Decaps  - decrypt the ciphertext with dk to recover m, then");
    println!("                 derive the same shared key K_A.");
    println!("  Success criterion: K_A == K_B for both Ladon128 and Ladon256.");
    println!();
    println!("  ek/dk/ciphertext are round-tripped through serialize/deserialize,");
    println!("  mirroring how these objects would travel on the wire.");
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
    println!("  {label} parameter set");
    println!("------------------------------------------------------------------");
    println!("    K   (module rank)        : {}", PARAMS::K);
    println!("    eta_1 (key CBD width)    : {}", PARAMS::ETA_1);
    println!("    eta_2 (noise CBD width)  : {}", PARAMS::ETA_2);
    println!("    d_u (u-compression bits) : {}", PARAMS::D_U);
    println!("    d_v (v-compression bits) : {}", PARAMS::D_V);
    println!("    q   (ring modulus)       : {Q}");
    println!("    N   (ring degree)        : 256");
    println!();

    // ---- 1) KeyGen --------------------------------------------------------
    // KeyGen samples the underlying K-PKE keypair and pairs it with the
    // FO finalization material (H(ek) and a fresh implicit-rejection z).
    let (ek, dk) = mlkem::key_gen::<PARAMS>();
    println!("[KeyGen] Generated encapsulation key (ek) and decapsulation key (dk).");
    println!("         ek is public; dk is private to the key holder.");

    // Round-trip ek/dk through serialize/deserialize.
    let ek_ser = ek.serialize();
    let dk_ser = dk.serialize();
    let ek_size_bytes = ek_ser.len() / 8;
    let dk_size_bytes = dk_ser.len() / 8;
    println!("         ek size on wire : {ek_size_bytes} bytes");
    println!("         dk size on wire : {dk_size_bytes} bytes");
    let ek = MlKemEncapsulationKey::<{ PARAMS::K }>::deserialize(&ek_ser);
    let dk = MlKemDecapsulationKey::<{ PARAMS::K }>::deserialize(&dk_ser);

    // ---- 2) Encaps --------------------------------------------------------
    // Asset owner samples a fresh 32-byte seed m, encrypts it under ek, and
    // derives the shared key K_B from G(m || H(ek)). The seed m never leaves
    // the encaps function; only the ciphertext and the derived key do.
    let (key_b, c) = mlkem::encaps::<PARAMS>(ek);
    let c_ser = c.serialize();
    let c_size_bytes = c_ser.len() / 8;
    println!();
    println!("[Encaps] Asset owner encapsulated a fresh 32-byte secret under ek.");
    println!("         ciphertext size : {c_size_bytes} bytes");
    println!("         K_B = {}", hex(&key_b));

    let c_des = MlKemCyphertext::<{ PARAMS::K }, { PARAMS::D_U }, { PARAMS::D_V }>::deserialize(
        &c_ser,
    );

    // ---- 3) Decaps --------------------------------------------------------
    // Key holder decrypts the ciphertext to recover m', re-derives the shared
    // key, and runs the FO check: re-encrypts m' under ek and compares with c.
    // On match, returns the derived key; on mismatch, returns J(z || c)
    // (implicit rejection - indistinguishable from a fresh random key to the
    // attacker, but never equal to K_B).
    let key_a = mlkem::decaps::<PARAMS>(c_des, dk);
    println!();
    println!("[Decaps] Key holder recovered the shared key from the ciphertext.");
    println!("         K_A = {}", hex(&key_a));

    // ---- Verdict ----------------------------------------------------------
    assert_eq!(
        key_a, key_b,
        "Ladon KEM round-trip failed for {label}: K_A != K_B"
    );
    println!();
    println!("[OK    ] K_A == K_B  - round-trip verified for {label}.");
}

fn hex(b: &[u8]) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    for x in b {
        let _ = write!(s, "{:02x}", x);
    }
    s
}