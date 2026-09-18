//! Bench: Ladon active encapsulation (asset-owner side).
//!
//! Times the encaps op:
//!   1. random 32-byte message m.
//!   2. hash(ek)  (precomputed once outside the loop; constant cost).
//!   3. G(m || H(ek)) -> (shared_key, encryption_randomness).
//!   4. kpke::encrypt_2k(ek, m, r) -> ciphertext.
//!
//! The hash H(ek) and z (implicit-rejection seed) are setup-only and not in
//! the timed path. This isolates the per-request cost the asset owner pays.
//!
//! Build & run:
//!     cargo run --release --bin bench_encapsulation_active

#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use std::hint::black_box;
use std::time::{Duration, Instant};

use bitvec::view::BitView;
use Ladon::crypt;
use Ladon::kpke;
use Ladon::params::*;
use Ladon::ring::{Compressed, Ring};
use Ladon::serialize::{BitOrder, MlKemDeserialize, MlKemSerialize};

// ===========================================================================
// ====== BENCH KNOBS — edit these ==========================================
// ===========================================================================
const ITERATIONS: usize = 1000;
const WARMUP: usize = 100;
// ===========================================================================

fn main() {
    println!("==================================================================");
    println!("  Ladon Bench: Active Encapsulation (asset-owner side)");
    println!("==================================================================");
    println!();
    println!("  Per iteration:");
    println!("    1. random m (32 bytes),");
    println!("    2. G(m || H(ek)) -> (key, rand),");
    println!("    3. kpke::encrypt_2k(ek, m, rand) -> ciphertext.");
    println!("  H(ek) is precomputed once outside the loop.");
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
    [(); 960 * PARAMS::K + 32]:,
{
    println!("------------------------------------------------------------------");
    println!("  {label}  (K = {})", PARAMS::K);
    println!("------------------------------------------------------------------");
    println!("    warmup / iter : {WARMUP} / {ITERATIONS}");
    println!();

    // ----- Setup (untimed): keygen + ek hash -------------------------------
    let (ek, _sk) = kpke::key_gen_2k::<PARAMS>();
    let ek_hash = crypt::h(&ek.serialize().into_vec());

    // ----- Warmup ----------------------------------------------------------
    for _ in 0..WARMUP {
        let out = encaps::<PARAMS>(&ek, &ek_hash);
        black_box(out);
    }

    // ----- Timed -----------------------------------------------------------
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let out = encaps::<PARAMS>(&ek, &ek_hash);
        black_box(out);
    }
    let total = start.elapsed();
    let avg = total / ITERATIONS as u32;
    let ops_per_sec = ITERATIONS as f64 / total.as_secs_f64();

    println!("    total    : {}", format_duration(total));
    println!("    avg / encpas : {}", format_duration(avg));
    println!("    ops / sec: {ops_per_sec:.2}");
}

#[inline(never)]
fn encaps<PARAMS: MlKemParams>(
    ek: &kpke::KpkeEncryptionKey<{ PARAMS::K }>,
    ek_hash: &[u8; 32],
) -> (
    [u8; 32],
    kpke::Cyphertext<{ PARAMS::K }, { PARAMS::D_U }, { PARAMS::D_V }>,
)
where
    [(); PARAMS::K]:,
    [(); PARAMS::ETA_1]:,
    [(); PARAMS::ETA_2]:,
    [(); 64 * PARAMS::ETA_1]:,
    [(); 64 * PARAMS::ETA_2]:,
    [(); 32 * (PARAMS::D_U * PARAMS::K + PARAMS::D_V)]:,
    [(); 960 * PARAMS::K + 32]:,
{
    let m_bytes = crypt::random_bytes::<32>();
    let mut combined = [0u8; 64];
    combined[..32].copy_from_slice(&m_bytes);
    combined[32..].copy_from_slice(ek_hash);
    let (key, rand) = crypt::g::<64>(&combined);

    let m: Compressed<1, Ring> =
        Compressed::<1, Ring>::deserialize(&m_bytes.view_bits::<BitOrder>().to_bitvec());
    let ct = kpke::encrypt_2k::<PARAMS>(ek.clone(), m, rand);
    (key, ct)
}

fn format_duration(d: Duration) -> String {
    // Always report in milliseconds so every timing field shares one unit.
    format!("{:.3} ms", d.as_nanos() as f64 / 1_000_000.0)
}