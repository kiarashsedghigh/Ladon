//! Bench: Active (SPDZ2k) TEE-side reconstruction + FO finalization.
//!
//! Per iteration (TEE-side):
//!   1. receiver_reconstruct: sum per-party <mu'*m>^{q'} shares, decode
//!      each of the ell candidates, majority-vote per coefficient, pack to
//!      32 bytes -> recovered message m.
//!   2. FO finalization: G(m || H(ek)) -> (key, rand); re-encrypt with
//!      kpke::encrypt_2k(ek, m, rand); compare to the asset owner's ct.
//!      On match return key; otherwise crypt::j(z || ct.serialize()).
//!
//! Sweeps n in {4, 8, 16, 32}; in active branch t = n - 1.
//!
//! Build & run:
//!     cargo run --release --bin bench_tee_active

#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use std::hint::black_box;
use std::time::{Duration, Instant};

use bitvec::view::BitView;
use rand::{rngs::StdRng, RngCore, SeedableRng};

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
// ====== BENCH KNOBS — edit these ==========================================
// ===========================================================================
const N_LIST: &[usize] = &[4, 8, 16, 32];
const ELL_LADON128: usize = 5;
const ELL_LADON256: usize = 32;
const S_BITS: u32 = 40;
const P_PLAINTEXT: u128 = 2;
const ITERATIONS: usize = 1500;
const WARMUP: usize = 200;
// ===========================================================================

fn main() {
    println!("==================================================================");
    println!("  Ladon Bench: Active (SPDZ2k) TEE-Side Reconstruction + FO");
    println!("==================================================================");
    println!();
    println!("  Per iteration:");
    println!("    1. receiver_reconstruct: sum <mu'*m>^{{q'}} shares, decode ell");
    println!("       candidates, majority-vote per coefficient, pack to 32 bytes.");
    println!("    2. FO finalization: G(m || H(ek)) -> (key, rand); re-encrypt;");
    println!("       compare against asset-owner ct; implicit rejection if mismatch.");
    println!();
    println!("  Convention: t = n - 1 (dishonest majority).");
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
    println!("    k_bits / s_bits : {K_BITS} / {S_BITS}");
    println!("    p               : {P_PLAINTEXT}");
    println!("    warmup / iter   : {WARMUP} / {ITERATIONS}");
    println!();
    println!(
        "  {:>5}    {:>5}    {:>14}    {:>14}    {:>12}",
        "t", "n", "total", "avg/op", "ops/sec"
    );
    println!(
        "  {:>5}    {:>5}    {:>14}    {:>14}    {:>12}",
        "-----", "-----", "--------------", "--------------", "------------"
    );

    for &n in N_LIST {
        bench_one::<PARAMS>(n, ell);
    }
}

fn bench_one<PARAMS: MlKemParams>(n: usize, ell: usize)
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
    // ----- SETUP (untimed) -------------------------------------------------
    let dealer = DealerSpdz::new(n, K_BITS, S_BITS, P_PLAINTEXT);
    let ks = dealer.generate_keypair::<PARAMS>();
    let dbl = dealer.generate_double_sharing(ell);

    // Asset-owner side: build a real ciphertext using the FO encaps path.
    let z = crypt::random_bytes::<32>();
    let ek_hash = crypt::h(&ks.ek.serialize().into_vec());

    let mut rng = StdRng::from_entropy();
    let mut m_bytes = [0u8; 32];
    rng.fill_bytes(&mut m_bytes);
    let mut combined = [0u8; 64];
    combined[..32].copy_from_slice(&m_bytes);
    combined[32..].copy_from_slice(&ek_hash);
    let (_key_a, rand) = crypt::g::<64>(&combined);

    let m: Compressed<1, Ring> =
        Compressed::<1, Ring>::deserialize(&m_bytes.view_bits::<BitOrder>().to_bitvec());
    let ct = kpke::encrypt_2k::<PARAMS>(ks.ek.clone(), m, rand);

    // Committee side (untimed): produce <mu'*m>^{q'} shares per party.
    let u_dec = decompress_vector_2k::<{ PARAMS::K }>(&ct.0 .0, PARAMS::D_U as u32, K_BITS);
    let v_dec = decompress_ring_2k(&ct.1 .0, PARAMS::D_V as u32, K_BITS);
    let parties: Vec<PartySpdz<{ PARAMS::K }>> =
        assemble_parties::<{ PARAMS::K }>(&ks.sk_shares, &dbl, dealer.thr);
    let mu_m_shares = threshold_decrypt(&parties, &u_dec, &v_dec, &dealer.thr);

    // ----- WARMUP ----------------------------------------------------------
    for _ in 0..WARMUP {
        let key = tee_pipeline::<PARAMS>(&mu_m_shares, &dealer.thr, &ks.ek, &ek_hash, &z, &ct);
        black_box(key);
    }

    // ----- TIMED -----------------------------------------------------------
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let key = tee_pipeline::<PARAMS>(&mu_m_shares, &dealer.thr, &ks.ek, &ek_hash, &z, &ct);
        black_box(key);
    }
    let total = start.elapsed();
    let avg = total / ITERATIONS as u32;
    let ops_per_sec = ITERATIONS as f64 / total.as_secs_f64();

    println!(
        "  {:>5}    {:>5}    {:>14}    {:>14}    {:>12.2}",
        n - 1,
        n,
        format_duration(total),
        format_duration(avg),
        ops_per_sec,
    );
}

#[inline(never)]
fn tee_pipeline<PARAMS: MlKemParams>(
    mu_m_shares: &[Vec<[u128; 256]>],
    thr: &Ladon::additive_2k::SpdzParams,
    ek: &kpke::KpkeEncryptionKey<{ PARAMS::K }>,
    ek_hash: &[u8; 32],
    z: &[u8; 32],
    ct: &kpke::Cyphertext<{ PARAMS::K }, { PARAMS::D_U }, { PARAMS::D_V }>,
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
    // Step 6: reconstruct m (with majority over ell).
    let m_bytes = receiver_reconstruct(mu_m_shares, thr);

    // FO finalization.
    let mut combined = [0u8; 64];
    combined[..32].copy_from_slice(&m_bytes);
    combined[32..].copy_from_slice(ek_hash);
    let (key_candidate, rand_prime) = crypt::g::<64>(&combined);

    let m: Compressed<1, Ring> =
        Compressed::<1, Ring>::deserialize(&m_bytes.view_bits::<BitOrder>().to_bitvec());
    let ct_prime = kpke::encrypt_2k::<PARAMS>(ek.clone(), m, rand_prime);

    if ct_prime.0 == ct.0 && ct_prime.1 == ct.1 {
        key_candidate
    } else {
        crypt::j([&z[..], ct.serialize().as_raw_slice()].concat())
    }
}

fn format_duration(d: Duration) -> String {
    let nanos = d.as_nanos();
    if nanos >= 1_000_000_000 {
        format!("{:.6} s", nanos as f64 / 1_000_000_000.0)
    } else if nanos >= 1_000_000 {
        format!("{:.3} ms", nanos as f64 / 1_000_000.0)
    } else if nanos >= 1_000 {
        format!("{:.3} us", nanos as f64 / 1_000.0)
    } else {
        format!("{nanos} ns")
    }
}