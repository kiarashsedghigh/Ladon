#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use bitvec::view::BitView;
use rand::{rngs::StdRng, RngCore, SeedableRng};

use Moiragus::{kpke, params::*};
use Moiragus::ring::{Compressed, Ring};
use Moiragus::serialize::{BitOrder, MlKemDeserialize, MlKemSerialize};

/// 32 cryptographically-random bytes. Mirrors `crypt::random_bytes::<32>()`,
/// reimplemented here because the `crypt` module is private to the crate.
fn random_32() -> [u8; 32] {
    let mut rng = StdRng::from_entropy();
    let mut out = [0u8; 32];
    rng.fill_bytes(&mut out);
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let paramset = args.get(1).map(|s| s.as_str()).unwrap_or("ML-KEM-512");

    match paramset {
        "ML-KEM-512" => pke_round_trip::<MlKem512>("ML-KEM-512"),
        "ML-KEM-768" => pke_round_trip::<MlKem768>("ML-KEM-768"),
        "ML-KEM-1024" => pke_round_trip::<MlKem1024>("ML-KEM-1024"),
        _ => panic!("Invalid parameter set: {paramset}"),
    };
}

fn pke_round_trip<PARAMS: MlKemParams>(label: &str)
where
    [(); 384 * PARAMS::K + 32]:,
    [(); 768 * PARAMS::K + 96]:,
    [(); PARAMS::K]:,
    [(); PARAMS::ETA_1]:,
    [(); PARAMS::ETA_2]:,
    [(); PARAMS::D_U]:,
    [(); 64 * PARAMS::ETA_1]:,
    [(); 64 * PARAMS::ETA_2]:,
    [(); 32 * (PARAMS::D_U * PARAMS::K + PARAMS::D_V)]:,
{
    println!("=== K-PKE round-trip: {label} ===");

    // ---- KeyGen -------------------------------------------------------------
    // K-PKE.KeyGen returns ((t, rho), s):
    //   ek_pke = (t, rho)  -- encryption key (public)
    //   dk_pke = s         -- decryption key (secret vector)
    let (ek_pke, dk_pke) = kpke::key_gen::<PARAMS>();
    println!("KeyGen done. ek = (t, rho), dk = s (secret vector of length {})", PARAMS::K);

    // ---- Message ------------------------------------------------------------
    // K-PKE encrypts exactly 32 bytes, carried as a `Compressed<1, Ring>`
    // (1 bit per coefficient => 256 bits = 32 bytes). We build it the same way
    // `mlkem::encaps` does: sample 32 random bytes, view as bits, byte_decode.
    let m_bytes = random_32();
    let m: Compressed<1, Ring> =
        Compressed::<1, Ring>::deserialize(&m_bytes.view_bits::<BitOrder>().to_bitvec());
    println!("Message  (32 bytes): {}", hex(&m_bytes));

    // ---- Encrypt ------------------------------------------------------------
    // K-PKE.Encrypt also needs 32 bytes of randomness `r`.
    let r = random_32();
    let c = kpke::encrypt::<PARAMS>(ek_pke, m, r);
    println!("Encrypt done. Ciphertext c = (u_compressed, v_compressed).");

    // ---- Decrypt ------------------------------------------------------------
    let m_prime: Compressed<1, Ring> = kpke::decrypt::<PARAMS>(dk_pke, c);

    // Serialize the recovered message back to 32 bytes to compare.
    let m_prime_bytes: Vec<u8> = m_prime.serialize().into_vec();
    println!("Recovered (32 bytes): {}", hex(&m_prime_bytes));

    assert_eq!(
        m_bytes.as_slice(),
        m_prime_bytes.as_slice(),
        "Decryption did not recover the original message!"
    );
    println!("Success! Recovered message matches the original.\n");
}

fn hex(b: &[u8]) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    for x in b {
        let _ = write!(s, "{:02x}", x);
    }
    s
}