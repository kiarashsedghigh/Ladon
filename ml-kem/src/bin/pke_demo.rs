//! K-PKE round-trip demo. Put this at `ml-kem/src/bin/pke_demo.rs`.
//!
//! Run with:
//!     cargo run --bin pke_demo
//!
//! Requires three small library edits (already applied if you've been
//! following along):
//!   * src/lib.rs:    `mod pke;`   → `pub mod pke;`
//!   * src/lib.rs:    `mod param;` → `pub mod param;`
//!   * src/pke.rs:    `pub(crate)` → `pub` on
//!                      DecryptionKey, EncryptionKey, generate, encrypt, decrypt
//!
//! What this does, mirroring `round_trip_test` in pke.rs:
//!   1. Seed d → K-PKE.KeyGen → (dk, ek)
//!   2. Pick message m and randomness r
//!   3. c = K-PKE.Encrypt(ek, m, r)
//!   4. m' = K-PKE.Decrypt(dk, c); assert m' == m
//!
//! Parameter sets: switch by changing the calls in main().
//!
//! Note on RNG: we deliberately don't pull in `rand_core::OsRng` or
//! `getrandom::SysRng` here, because the rand_core/getrandom API has
//! churned across versions and the exact names depend on which version
//! `ml-kem`'s Cargo.lock resolved to. For a *correctness* round-trip
//! demo, the bytes just need to be different on each run — they don't
//! need to be cryptographically random. So we mix the current time
//! into 32 bytes using a tiny SplitMix64 expansion. Zero extra deps.
//!
//! Note on modulus q: q = 3329 is hardcoded in src/algebra.rs
//! (define_field!, ZETA = 17 tables, and Elem::new(3303) = 128⁻¹ mod 3329
//! in ntt_inverse). Changing q requires editing those three places
//! together — can't be done from a binary.

#![allow(non_snake_case)]

use std::time::{SystemTime, UNIX_EPOCH};

use ml_kem::param::PkeParams;
use ml_kem::pke::{DecryptionKey, EncryptionKey};
use ml_kem::{B32, MlKem512, MlKem768, MlKem1024};

/// Tiny PRNG. SplitMix64 is a well-known, public-domain mixer; we use it
/// purely to spread one u64 seed across a 32-byte buffer so each run of
/// the demo uses different inputs. DO NOT use this construction as a
/// cryptographic RNG in real code.
struct SplitMix64(u64);
impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn fill(&mut self, out: &mut [u8]) {
        for chunk in out.chunks_mut(8) {
            let bytes = self.next_u64().to_le_bytes();
            chunk.copy_from_slice(&bytes[..chunk.len()]);
        }
    }
}

fn seed_from_time() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0xDEAD_BEEF_CAFE_BABE)
}

fn rand_b32(rng: &mut SplitMix64) -> B32 {
    let mut out = B32::default();
    rng.fill(out.as_mut_slice());
    out
}

fn round_trip<P: PkeParams>(label: &str, rng: &mut SplitMix64) {
    // ---- KeyGen ----
    let d = rand_b32(rng);
    let (dk, ek): (DecryptionKey<P>, EncryptionKey<P>) = DecryptionKey::<P>::generate(&d);

    // ---- Encrypt ----
    let msg: B32 = rand_b32(rng);
    let r:   B32 = rand_b32(rng);
    let ct = ek.encrypt(&msg, &r);

    // ---- Decrypt ----
    let recovered: B32 = dk.decrypt(&ct);

    print!("[{label:>9}] msg = {}… ct = {} bytes … ", hex_prefix(&msg, 8), ct.len());
    if recovered == msg {
        println!("OK");
    } else {
        println!("MISMATCH");
        println!("           got = {}", hex_prefix(&recovered, 32));
        std::process::exit(1);
    }
}

fn main() {
    let mut rng = SplitMix64(seed_from_time());
    println!("K-PKE round-trip (KeyGen → Encrypt → Decrypt)\n");
    round_trip::<MlKem512>("MlKem512", &mut rng);
    round_trip::<MlKem768>("MlKem768", &mut rng);
    round_trip::<MlKem1024>("MlKem1024", &mut rng);
}

fn hex_prefix(b: &[u8], n: usize) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    for x in &b[..n.min(b.len())] {
        let _ = write!(s, "{:02x}", x);
    }
    s
}