//! Demo: Ladon threshold KEM round-trip.
//!
//! Replaces the single-key-holder decapsulation of the centralized demo with
//! a committee-coordinated one. The asset owner's encapsulation is unchanged.
//!
//! Threshold convention used throughout this file:
//!   t = THRESHOLD = the minimum size of a decrypting set.
//!     - any t (or more) parties can cooperate to decrypt.
//!     - any t-1 parties cannot decrypt.
//!   The underlying Shamir polynomial has degree t-1, so when calling
//!   Dealer::new (which takes the polynomial degree as its first argument)
//!   we pass THRESHOLD - 1.
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
const N_PARTIES: usize = 9;     // total parties the dealer shares to
const THRESHOLD: usize = 4;     // t : ANY t (or more) parties can decrypt;
//     any t-1 cannot.
const P_PLAINTEXT: u64 = 2;     // plaintext modulus (power of two)

// Parallel double sharings per security level (Section 5.1 of the paper).
// Receiver runs ell parallel finalizations and takes coefficient-wise
// majority across the ell candidate decodes. Larger ell -> lower decryption-
// failure probability, more work per party in mask + finalize.
const ELL_LADON128: usize = 5;
const ELL_LADON256: usize = 32;

// Committee members (0-based indices; need >= THRESHOLD of them).
const ACTIVE: &[usize] = &[0, 1, 2, 3, 4, 5, 6, 7, 8];
// Both Ladon128 and Ladon256 are run back-to-back from main().
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
    println!("==================================================================");
    println!("  Ladon threshold KEM round-trip demo");
    println!("==================================================================");
    println!();
    println!("  What this demo does:");
    println!("    1. KeyGen     - dealer generates (ek, dk) and secret-shares dk");
    println!("                    across n parties so any t (or more) can");
    println!("                    cooperate to decapsulate. The TEE keeps the");
    println!("                    FO finalization material (H(ek) and a fresh z).");
    println!("    2. Encaps     - the asset owner encapsulates a 32-byte secret");
    println!("                    under ek, identical to the centralized case.");
    println!("    3. T-Decaps   - the active committee (>= t parties) runs the");
    println!("                    distributed decryption protocol with ell-fold");
    println!("                    majority decoding to recover the seed m'. The");
    println!("                    TEE then runs the FO finalization locally.");
    println!("  Success criterion: K_A_threshold == K_B (asset key derived by");
    println!("  the asset owner).");
    println!();
    run::<Ladon128>("Ladon128", ELL_LADON128);
    println!();
    run::<Ladon256>("Ladon256", ELL_LADON256);
    println!("==================================================================");
}

/// Threshold decapsulation. Body mirrors `mlkem::decaps`, except that the
/// K-PKE decryption step is replaced by the threshold protocol followed by
/// ell-fold majority decoding (see `Ladon::threshold`).
///
/// In Ladon deployment this is the TEE's role: coordinate the distributed
/// decryption with the KBS committee to recover the seed m', then run the FO
/// finalization (G, re-encrypt, compare against c, implicit rejection on
/// mismatch) locally inside the enclave. The output is the asset key.
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
    // Committee runs Steps 1-3 to produce per-party (Phi*m)_j shares.
    let phi_m_shares = threshold_decrypt(parties, c.clone(), thr);
    // TEE runs Step 4 + ell-majority decoding locally to recover the seed.
    let m_bytes: [u8; 32] = receiver_reconstruct(&phi_m_shares, thr);

    // Seed bytes -> Compressed<1, Ring> for re-encryption.
    let m: Compressed<1, Ring> =
        Compressed::<1, Ring>::deserialize(&m_bytes.view_bits::<BitOrder>().to_bitvec());

    // FO finalization: G(m' || H(ek)) -> (K', r')
    let mut combined = [0u8; 64];
    combined[..32].copy_from_slice(&m_bytes);
    combined[32..].copy_from_slice(&hash);
    let (key, rand) = crypt::g::<64>(&combined);

    // Re-encrypt under ek with derived randomness and compare.
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
    assert!(
        ACTIVE.len() >= THRESHOLD,
        "need at least t = {} active parties, got {}",
        THRESHOLD,
        ACTIVE.len()
    );
    assert!(ell >= 1, "ell must be >= 1");
    assert!(THRESHOLD >= 1, "THRESHOLD must be >= 1");

    println!("------------------------------------------------------------------");
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
    println!("    n     (total parties)       : {N_PARTIES}");
    println!("    t     (threshold)           : {THRESHOLD}    (any {THRESHOLD} can decrypt; {} cannot)",
             THRESHOLD - 1);
    println!("    active committee            : {ACTIVE:?}");
    println!("    ell   (parallel sharings)   : {ell}    (majority-decoded at TEE)");
    println!("    p     (plaintext modulus)   : {P_PLAINTEXT}");
    println!();

    // ---- 1) Dealer keygen: ek + sk shares + ell double sharings ----------
    println!("[KeyGen] Dealer is generating ek + secret-key shares for {N_PARTIES}");
    println!("         parties (degree-{} Shamir polynomial; need {THRESHOLD} shares to", THRESHOLD - 1);
    println!("         reconstruct). Also pre-computing {ell} parallel double");
    println!("         sharings used in the decryption protocol.");
    // Dealer takes the Shamir polynomial degree as its first arg.
    // Under our convention t = min-to-decrypt, so polynomial degree = t-1.
    let dealer = Dealer::new(THRESHOLD - 1, N_PARTIES, P_PLAINTEXT);
    println!("         Threshold ring constants:");
    println!("           q     = ring modulus           = {}", dealer.thr.q);
    println!("           q'    = 2^round(log2(q))       = {}", dealer.thr.q_prime);
    println!("           mu    = q / p                  = {}", dealer.thr.mu);
    println!("           mu'   = q' / p                 = {}", dealer.thr.mu_prime);
    let ks = dealer.generate_keypair::<PARAMS>();
    let dbl = dealer.generate_double_sharing(ell);
    println!(
        "         Done. {} sk shares + {} double sharings per party.",
        ks.sk_shares.len(),
        ell
    );

    // FO finalization material kept by the TEE alongside ek.
    let hash = crypt::h(&ks.ek.clone().serialize().into_vec());
    let z = random_32();
    println!("         TEE generated FO material: hash = H(ek), z = {} (random)", hex(&z));

    // Sanity: two disjoint quorums of size THRESHOLD must reconstruct the
    // same secret key.
    let rec1 = reconstruct_vector(&ks.sk_shares[0..THRESHOLD].to_vec());
    let rec2 = reconstruct_vector(&ks.sk_shares[N_PARTIES - THRESHOLD..N_PARTIES].to_vec());
    assert_eq!(rec1, rec2, "key shares inconsistent");
    println!("         [check] two disjoint quorums of size {THRESHOLD} reconstruct identical key. OK");
    println!();

    // ---- 2) Asset owner: encaps under ek ---------------------------------
    println!("[Encaps] Asset owner samples a fresh 32-byte secret seed, encapsulates");
    println!("         it under the committee's public key (ek), and derives the");
    println!("         asset key K_B = G(seed, H(ek)). Ciphertext c is sent to the");
    println!("         TEE for decapsulation.");
    let (key_b, c) = mlkem::encaps::<PARAMS>(ks.ek.clone());
    println!("         K_B (asset key)                = {}", hex(&key_b));
    println!();

    // ---- 3) Threshold decapsulation --------------------------------------
    println!("[T-Decaps] {} parties cooperate to decrypt c:", ACTIVE.len());
    println!("         each party computes its share of (Phi * m) locally and");
    println!("         the TEE reconstructs the seed via {ell}-fold majority");
    println!("         decoding, then runs the FO finalization (G + re-encrypt");
    println!("         + compare; J(z || c) on mismatch) inside the enclave.");
    let parties: Vec<Party<{ PARAMS::K }>> =
        assemble_parties::<{ PARAMS::K }>(&ks.sk_shares, &dbl, ACTIVE, dealer.thr);
    let key_a_threshold = threshold_decaps::<PARAMS>(
        c.clone(),
        &parties,
        &dealer.thr,
        ks.ek.clone(),
        hash,
        z,
    );
    println!("         K_A (threshold)                = {}", hex(&key_a_threshold));

    // ---- 4) Verdict ------------------------------------------------------
    println!();
    if key_a_threshold == key_b {
        println!("[OK    ] K_A_threshold == K_B for {label}.");
        println!("         Threshold round-trip verified end-to-end.");
    } else {
        println!("[FAIL  ] K_A_threshold != K_B for {label}.");
        println!("         Threshold protocol failed to recover the asset key.");
    }

    assert_eq!(
        key_a_threshold, key_b,
        "Ladon threshold round-trip failed for {label}: K_A_threshold != K_B"
    );
}