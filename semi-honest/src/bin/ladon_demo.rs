//! Demo: Ladon threshold KEM round-trip.
//!
//! Threshold convention:
//!   t = min-to-decrypt = active-committee size.
//!     - The first t parties (ids 1..=t) form the active committee.
//!     - The dealer creates t additive double sharings, one per committee
//!       member. Additive sharing is intrinsically full-set: all t members
//!       must participate to reconstruct the masking polynomial.
//!     - The dealer also creates n Shamir key shares (any t reconstruct sk),
//!       providing offline robustness against corruption of up to t-1
//!       long-term shareholders.
//!
//! Build & run:
//!     cargo run --release --bin demo_threshold_kem

#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use bitvec::view::BitView;
use rand::{rngs::StdRng, RngCore, SeedableRng};

use Ladon::crypt;
use Ladon::dealer::{Dealer, ThrParams};
use Ladon::kpke;
use Ladon::mlkem::{self, MlKemCyphertext, MlKemEncapsulationKey};
use Ladon::params::*;
use Ladon::party::Party;
use Ladon::ring::{Compressed, Ring};
use Ladon::serialize::{BitOrder, MlKemDeserialize, MlKemSerialize};
use Ladon::shamir_poly_ring::reconstruct_vector;
use Ladon::threshold::{assemble_parties, receiver_reconstruct, threshold_decrypt};

// ===========================================================================
// ====== DEMO PARAMETERS — edit these ======================================
// ===========================================================================
const N_PARTIES: usize = 9;     // total Shamir shareholders (offline robustness)
const THRESHOLD: usize = 4;     // t: active-committee size = min-to-decrypt
const P_PLAINTEXT: u64 = 2;

const ELL_LADON128: usize = 5;
const ELL_LADON256: usize = 32;
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
    for x in b { let _ = write!(s, "{:02x}", x); }
    s
}

fn main() {
    println!("==================================================================");
    println!("  Ladon threshold KEM round-trip demo");
    println!("==================================================================");
    println!();
    println!("  What this demo does:");
    println!("    1. KeyGen     - dealer generates (ek, dk). dk is Shamir-shared");
    println!("                    across n long-term shareholders (any t can");
    println!("                    reconstruct), and t additive double sharings");
    println!("                    are produced for the active committee (the");
    println!("                    FIRST t party ids).");
    println!("    2. Encaps     - the asset owner encapsulates a 32-byte secret");
    println!("                    under ek, identical to the centralized case.");
    println!("    3. T-Decaps   - the t active committee members run the");
    println!("                    distributed decryption protocol with ell-fold");
    println!("                    majority decoding to recover the seed m'. The");
    println!("                    TEE runs the FO finalization locally.");
    println!();
    run::<Ladon128>("Ladon128", ELL_LADON128);
    println!();
    run::<Ladon256>("Ladon256", ELL_LADON256);
    println!("==================================================================");
}

fn threshold_decaps<PARAMS: MlKemParams>(
    c: MlKemCyphertext<{ PARAMS::K }, { PARAMS::D_U }, { PARAMS::D_V }>,
    parties: &[Party<{ PARAMS::K }>],
    thr: &ThrParams,
    ek: MlKemEncapsulationKey<{ PARAMS::K }>,
    hash: [u8; 32],
    z: [u8; 32],
) -> [u8; 32]
where
    [(); 960 * PARAMS::K + 32]:,
    [(); PARAMS::K]:,
    [(); PARAMS::ETA_1]:,
    [(); PARAMS::ETA_2]:,
    [(); 64 * PARAMS::ETA_1]:,
    [(); 64 * PARAMS::ETA_2]:,
    [(); 32 * (PARAMS::D_U * PARAMS::K + PARAMS::D_V)]:,
{
    let mu_m_shares = threshold_decrypt(parties, c.clone(), thr);
    let m_bytes: [u8; 32] = receiver_reconstruct(&mu_m_shares, thr);

    let m: Compressed<1, Ring> =
        Compressed::<1, Ring>::deserialize(&m_bytes.view_bits::<BitOrder>().to_bitvec());

    let mut combined = [0u8; 64];
    combined[..32].copy_from_slice(&m_bytes);
    combined[32..].copy_from_slice(&hash);
    let (key, rand) = crypt::g::<64>(&combined);

    let c_prime = kpke::encrypt::<PARAMS>(ek, m, rand);
    if c.0 == c_prime.0 && c.1 == c_prime.1 {
        key
    } else {
        crypt::j([&z, c.serialize().as_raw_slice()].concat())
    }
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
{
    assert!(ell >= 1, "ell must be >= 1");
    assert!(THRESHOLD >= 2, "THRESHOLD must be >= 2");
    assert!(N_PARTIES >= THRESHOLD, "N_PARTIES must be >= THRESHOLD");

    println!("\n\n------------------------------------------------------------------");
    println!("  {label} parameter set");
    println!("------------------------------------------------------------------");
    println!("  Lattice parameters:");
    println!("    K     (module rank)         : {}", PARAMS::K);
    println!("    eta_1 (key CBD width)       : {}", PARAMS::ETA_1);
    println!("    eta_2 (noise CBD width)     : {}", PARAMS::ETA_2);
    println!("    d_u   (u-compression bits)  : {}", PARAMS::D_U);
    println!("    d_v   (v-compression bits)  : {}", PARAMS::D_V);
    println!("    q     (ring modulus)        : {Q}");
    println!("    N     (ring degree)         : 256");
    println!();
    println!("  Threshold parameters:");
    println!("    n     (Shamir shareholders) : {N_PARTIES}");
    println!("    t     (active committee)    : {THRESHOLD}    (first {THRESHOLD} party ids)");
    println!("    ell   (parallel sharings)   : {ell}    (majority-decoded at TEE)");
    println!("    p     (plaintext modulus)   : {P_PLAINTEXT}");
    println!();

    // ---- 1) Dealer keygen ------------------------------------------------
    println!("[KeyGen] Dealer generates ek + n={N_PARTIES} Shamir key shares");
    println!("         (degree-{} polynomial; any {THRESHOLD} shares reconstruct sk),", THRESHOLD - 1);
    println!("         plus {ell} parallel additive double sharings across the");
    println!("         t={THRESHOLD} active-committee members (party ids 1..={THRESHOLD}).");
    let dealer = Dealer::new(THRESHOLD, N_PARTIES, P_PLAINTEXT);
    println!("         Threshold ring constants:");
    println!("           q     = {}", dealer.thr.q);
    println!("           q'    = {}", dealer.thr.q_prime);
    println!("           mu    = {}", dealer.thr.mu);
    println!("           mu'   = {}", dealer.thr.mu_prime);
    let ks = dealer.generate_keypair::<PARAMS>();
    let dbl = dealer.generate_double_sharing(ell);
    println!(
        "         Done. {} Shamir sk shares + {} active-committee double shares ({} each).",
        ks.sk_shares.len(),
        dbl.len(),
        ell
    );

    let hash = crypt::h(&ks.ek.clone().serialize().into_vec());
    let z = random_32();
    println!("         TEE generated FO material: hash = H(ek), z = {} (random)", hex(&z));

    // Sanity: two disjoint t-subsets of Shamir shares reconstruct the same sk.
    let rec1 = reconstruct_vector(&ks.sk_shares[0..THRESHOLD].to_vec());
    let rec2 = reconstruct_vector(&ks.sk_shares[N_PARTIES - THRESHOLD..N_PARTIES].to_vec());
    assert_eq!(rec1, rec2, "key shares inconsistent");
    println!("         [check] two disjoint Shamir quorums of size {THRESHOLD} reconstruct identical sk. OK");
    println!();

    // ---- 2) Asset owner: encaps under ek ---------------------------------
    println!("[Encaps] Asset owner encapsulates a fresh 32-byte secret under ek");
    println!("         and derives the asset key K_B = G(seed, H(ek)).");
    let (key_b, c) = mlkem::encaps::<PARAMS>(ks.ek.clone());
    println!("         K_B (asset key)                = {}", hex(&key_b));
    println!();

    // ---- 3) Threshold decapsulation --------------------------------------
    println!("[T-Decaps] The {} active committee members cooperate to decrypt c;", THRESHOLD);
    println!("         the TEE runs {ell}-fold majority decoding and the FO");
    println!("         finalization locally inside the enclave.");
    let parties: Vec<Party<{ PARAMS::K }>> =
        assemble_parties::<{ PARAMS::K }>(&ks.sk_shares, &dbl, dealer.thr);
    let key_a_threshold = threshold_decaps::<PARAMS>(
        c.clone(),
        &parties,
        &dealer.thr,
        ks.ek.clone(),
        hash,
        z,
    );
    println!("         K_A (threshold)                = {}", hex(&key_a_threshold));

    println!();
    if key_a_threshold == key_b {
        println!("[OK    ] K_A_threshold == K_B for {label}.");
    } else {
        println!("[FAIL  ] K_A_threshold != K_B for {label}.");
    }
    assert_eq!(
        key_a_threshold, key_b,
        "Ladon threshold round-trip failed for {label}"
    );
}