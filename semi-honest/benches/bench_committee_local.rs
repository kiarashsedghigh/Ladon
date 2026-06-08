//! Bench: Ladon KBS Committee Local Computation (per-party, no network).
//!
//! Times the local compute each KBS authority performs during distributed
//! decryption (Figure 2 of the paper). Network rounds (opening, sending
//! to TEE) are intentionally excluded - they would dominate end-to-end
//! latency but are not "compute cost".
//!
//! Threshold convention: t = THRESHOLD = min-to-decrypt. Polynomial degree
//! = t - 1. SETUP_T is fixed because per-party cost is essentially
//! t-independent (partial_decrypt is a constant K-ring inner product).
//!
//! Ell is fixed per security level (matching the demo and bench_tee), not
//! swept - the per-party cost is the SUM of:
//!   - partial_decrypt: ell-independent, dominant.
//!   - mask + finalize: linear in ell, small.
//! Each security level runs once at its target ell.
//!
//! Build & run:
//!     cargo run --release --bin bench_committee_local

#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use std::hint::black_box;
use std::time::{Duration, Instant};

use Ladon::dealer::{Dealer, ThrParams};
use Ladon::mlkem;
use Ladon::params::*;
use Ladon::party::Party;
use Ladon::threshold::{assemble_parties, open_v, threshold_decrypt};

// ===========================================================================
// ====== BENCH KNOBS — edit these ==========================================
// ===========================================================================
// Parallel double sharings per security level (Section 5.1 of the paper).
const ELL_LADON128: usize = 5;
const ELL_LADON256: usize = 32;

const P_PLAINTEXT: u64 = 2;
// Threshold convention: t = min-to-decrypt. Polynomial degree = t-1.
// SETUP_T is fixed because per-party cost is essentially t-independent.
const SETUP_T: usize = 4;
const SETUP_N: usize = 9;        // satisfies SETUP_T < SETUP_N / 2
const ITERATIONS: usize = 5000;
const WARMUP: usize = 500;
// ===========================================================================

fn main() {
    println!("==================================================================");
    println!("  Ladon Bench: KBS Committee Local Computation (per-party)");
    println!("==================================================================");
    println!();
    println!("  What this bench measures:");
    println!("    For one KBS party, the local compute spent on a single");
    println!("    decryption request. The pipeline per iteration is:");
    println!("      1. partial_decrypt - inner product against the local secret-");
    println!("         key share, mod-switch from q to eta, isolate noise mod Phi.");
    println!("      2. mask            - add the party's ell shares of the");
    println!("         masking polynomial r to its e' share.");
    println!("      3. finalize        - given the (already opened) v = e' + r,");
    println!("         compute the party's ell shares of Phi * m.");
    println!();
    println!("  EXCLUDED (these are network rounds, not local compute):");
    println!("      - broadcasting v shares and reconstructing v = e' + r,");
    println!("      - sending the (Phi * m) shares to the TEE.");
    println!();
    println!("  Per-party cost is essentially t-independent (partial_decrypt is");
    println!("  a constant K-ring inner product). Mask + finalize loop over ell");
    println!("  and contribute the ell-dependent slope - small compared to");
    println!("  partial_decrypt. Multiply avg/op by t for the aggregate sequential");
    println!("  committee compute on a single machine.");
    println!();
    println!("  Convention: t = min-to-decrypt.");
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
{
    println!("------------------------------------------------------------------");
    println!("  {label} parameter set");
    println!("------------------------------------------------------------------");
    println!("    K     (module rank)         : {}", PARAMS::K);
    println!("    eta_1 (key CBD width)       : {}", PARAMS::ETA_1);
    println!("    eta_2 (noise CBD width)     : {}", PARAMS::ETA_2);
    println!("    d_u   (u-compression bits)  : {}", PARAMS::D_U);
    println!("    d_v   (v-compression bits)  : {}", PARAMS::D_V);
    println!("    q     (ring modulus)        : {Q}");
    println!("    N     (ring degree)         : 256");
    println!("    p     (plaintext modulus)   : {P_PLAINTEXT}");
    println!("    t / n                       : {SETUP_T} / {SETUP_N}    (t = min-to-decrypt)");
    println!("    ell   (parallel sharings)   : {ell}");
    println!("    warmup / timed iterations   : {WARMUP} / {ITERATIONS}");
    println!();

    // ===== SETUP (untimed) ================================================
    // SETUP_T = min-to-decrypt; polynomial degree = SETUP_T - 1.
    let dealer = Dealer::new(SETUP_T - 1, SETUP_N, P_PLAINTEXT);
    let ks = dealer.generate_keypair::<PARAMS>();
    let dbl = dealer.generate_double_sharing(ell);

    let (_key_b, c) = mlkem::encaps::<PARAMS>(ks.ek.clone());

    // Active committee = first SETUP_T parties (smallest decrypting set).
    // parties[0] has the smallest Shamir x-value (= 1), so it is the
    // v-holder; we'll time parties[1] (a non-v-holder).
    let active: Vec<usize> = (0..SETUP_T).collect();
    let parties: Vec<Party<{ PARAMS::K }>> =
        assemble_parties::<{ PARAMS::K }>(&ks.sk_shares, &dbl, &active, dealer.thr);
    let active_ids: Vec<u32> = parties.iter().map(|p| p.id).collect();
    let v_holder_id = *active_ids.iter().min().expect("need >=1 party");

    // Precompute the opened v_j (simulating the network round once).
    let step1_all: Vec<_> = parties
        .iter()
        .map(|p| {
            let is_holder = p.id == v_holder_id;
            p.partial_decrypt_compressed::<{ PARAMS::D_U }, { PARAMS::D_V }>(
                c.clone(),
                &active_ids,
                is_holder,
            )
        })
        .collect();
    let masked_all: Vec<Vec<[u64; 256]>> = parties
        .iter()
        .zip(step1_all.iter())
        .map(|(p, out)| p.mask(&out.e_prime))
        .collect();
    let v_opened: Vec<[u64; 256]> = open_v(&masked_all, &dealer.thr);

    // Sanity: end-to-end protocol still works for this ell.
    {
        let phi_m_shares = threshold_decrypt(&parties, c.clone(), &dealer.thr);
        assert_eq!(phi_m_shares[0].len(), ell);
        black_box(phi_m_shares);
    }

    let target = &parties[1]; // non-v-holder (common case)
    let is_holder = target.id == v_holder_id;
    debug_assert!(!is_holder);

    // ===== WARMUP =========================================================
    for _ in 0..WARMUP {
        let k = local_pipeline::<PARAMS>(target, &c, &active_ids, &dealer.thr, &v_opened, is_holder);
        black_box(k);
    }

    // ===== TIMED LOOP =====================================================
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let k = local_pipeline::<PARAMS>(target, &c, &active_ids, &dealer.thr, &v_opened, is_holder);
        black_box(k);
    }
    let total = start.elapsed();
    let avg = total / ITERATIONS as u32;
    let ops_per_sec = ITERATIONS as f64 / total.as_secs_f64();

    println!("    total time     : {}", format_duration(total));
    println!("    avg / party-op : {}", format_duration(avg));
    println!("    ops / sec      : {:.2}", ops_per_sec);
}

#[inline(never)]
fn local_pipeline<PARAMS: MlKemParams>(
    party: &Party<{ PARAMS::K }>,
    c: &Ladon::mlkem::MlKemCyphertext<{ PARAMS::K }, { PARAMS::D_U }, { PARAMS::D_V }>,
    active_ids: &[u32],
    thr: &ThrParams,
    v_opened: &[[u64; 256]],
    is_holder: bool,
) -> Vec<[u64; 256]>
where
    [(); PARAMS::K]:,
    [(); 32 * (PARAMS::D_U * PARAMS::K + PARAMS::D_V)]:,
{
    let _ = thr;

    let step1 =
        party.partial_decrypt_compressed::<{ PARAMS::D_U }, { PARAMS::D_V }>(
            c.clone(),
            active_ids,
            is_holder,
        );

    // Mask: in deployment the masked share is broadcast - that's network and
    // not in the timed budget. Use black_box so the optimizer keeps the work.
    let masked = party.mask(&step1.e_prime);
    black_box(&masked);

    party.finalize(&step1.w_prime, v_opened, is_holder)
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