//! Bench: Broadcast / open communication only (paramset-agnostic).
//!
//! Measures the all-to-all broadcast in the middle of threshold decryption.
//! Each party sends `payload_bytes` to every other party and waits to
//! receive theirs. Per-party local compute (mask, finalize, ...) and the
//! ciphertext / partial-result rounds are measured by other benches; this
//! file is exclusively about the open step's network cost.
//!
//! Knobs are set directly in bytes — payload size is the user's choice and
//! is NOT derived from ell or any cryptographic parameter.
//!
//! Network model
//! -------------
//! Symmetric one-way propagation latency L (ms) and per-party uplink
//! bandwidth BW (Mbps). Each party serializes its n-1 sends on its own
//! uplink, so the k-th recipient gets the message at time L + k*(S/B). The
//! broadcast wall-clock is therefore approximately
//!     L + (n-1) * (S / (BW / 8))
//! plus mpsc + scheduling noise.
//!
//! BW_mbps = 0 means "infinite bandwidth": only latency is modeled.
//!
//! 16-core machine, n up to ~32 is fine. Sleeping threads consume no CPU;
//! the contended phases (sends + receives) are microseconds. Above n ~ 64
//! the OS scheduler shows up in the numbers.
//!
//! Build & run:
//!     cargo run --release --bin bench_comm

use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

// ===========================================================================
// ====== BENCH KNOBS — edit these ==========================================
// ===========================================================================
/// One-way network latencies to sweep (milliseconds).
const NETWORK_LATENCIES_MS: &[u64] = &[1, 15];

/// Per-party uplink bandwidth in Mbps. 0 = infinite (latency-only model).
const NETWORK_BANDWIDTH_MBPS: u64 = 1000;

/// Committee sizes to sweep.
const N_PARTIES_LIST: &[usize] = &[4, 8, 16];

/// Per-message payload sizes in BYTES. Each party broadcasts a buffer of
/// this many bytes to every other party. Specify whatever values you want
/// to bench — independent of ell, K, or any cryptographic parameter.
const PAYLOAD_SIZES_BYTES: &[usize] = &[(14.23 as usize) * 1024];

/// Timed iterations per (payload, n, latency).
const ITERATIONS: usize = 1000;
/// Warmup iterations per (payload, n, latency).
const WARMUP: usize = 100;
// ===========================================================================

fn main() {
    println!("==================================================================");
    println!("  Bench: Broadcast Communication (open step only)");
    println!("==================================================================");
    println!();
    println!("  What this bench measures:");
    println!("    The all-to-all broadcast where each of n parties sends a buffer");
    println!("    of S bytes to every other party. EXCLUDED:");
    println!("      - asset owner -> committee ciphertext delivery,");
    println!("      - committee   -> TEE   partial-result delivery.");
    println!();
    println!("  Network model:");
    println!("    Symmetric one-way latency L (ms) and per-party uplink BW (Mbps).");
    println!("    Each party serializes its n-1 sends on its own uplink, so the");
    println!("    k-th recipient gets the message at time L + k*(S/B).");
    println!();
    println!("  Knobs:");
    println!("    PAYLOAD_SIZES_BYTES   = {:?}", PAYLOAD_SIZES_BYTES);
    println!("    N_PARTIES_LIST        = {:?}", N_PARTIES_LIST);
    println!("    NETWORK_LATENCIES_MS  = {:?}", NETWORK_LATENCIES_MS);
    println!("    NETWORK_BANDWIDTH_MBPS= {NETWORK_BANDWIDTH_MBPS}    (0 = infinite)");
    println!("    warmup / iterations   = {WARMUP} / {ITERATIONS}");
    println!();

    for &payload_bytes in PAYLOAD_SIZES_BYTES {
        run_for_payload(payload_bytes);
        println!();
    }
    println!("==================================================================");
}

fn run_for_payload(payload_bytes: usize) {
    // Build the per-party payload ONCE per payload size, sized for the max n.
    let n_max = *N_PARTIES_LIST.iter().max().expect("N_PARTIES_LIST empty");
    let payload_per_party: Vec<Vec<u8>> =
        (0..n_max).map(|_| vec![0u8; payload_bytes]).collect();

    println!("------------------------------------------------------------------");
    println!("  Payload: {} per message", format_bytes(payload_bytes));
    println!("------------------------------------------------------------------");
    if NETWORK_BANDWIDTH_MBPS == 0 {
        println!("    uplink bandwidth           : infinite (latency-only)");
    } else {
        let tx_per_msg = transmission_per_msg(payload_bytes, NETWORK_BANDWIDTH_MBPS);
        println!(
            "    uplink bandwidth           : {NETWORK_BANDWIDTH_MBPS} Mbps  ({} / msg)",
            format_duration(tx_per_msg)
        );
    }
    println!();
    println!(
        "  {:>5}    {:>10}    {:>14}    {:>14}    {:>12}",
        "n", "latency", "total", "avg/op", "ops/sec"
    );
    println!(
        "  {:>5}    {:>10}    {:>14}    {:>14}    {:>12}",
        "---", "----------", "--------------", "--------------", "------------"
    );

    for &n in N_PARTIES_LIST {
        for &latency_ms in NETWORK_LATENCIES_MS {
            bench_one(
                &payload_per_party[..n],
                n,
                latency_ms,
                NETWORK_BANDWIDTH_MBPS,
                payload_bytes,
            );
        }
    }
}

fn bench_one(
    payloads: &[Vec<u8>],
    n: usize,
    latency_ms: u64,
    bandwidth_mbps: u64,
    payload_bytes: usize,
) {
    debug_assert_eq!(payloads.len(), n);

    let latency = Duration::from_millis(latency_ms);
    let tx_per_msg = transmission_per_msg(payload_bytes, bandwidth_mbps);

    for _ in 0..WARMUP {
        run_broadcast(n, payloads, latency, tx_per_msg);
    }

    let start = Instant::now();
    for _ in 0..ITERATIONS {
        run_broadcast(n, payloads, latency, tx_per_msg);
    }
    let total = start.elapsed();
    let avg = total / ITERATIONS as u32;
    let ops_per_sec = ITERATIONS as f64 / total.as_secs_f64();

    println!(
        "  {:>5}    {:>10}    {:>14}    {:>14}    {:>12.2}",
        n,
        format!("{latency_ms} ms"),
        format_duration(total),
        format_duration(avg),
        ops_per_sec,
    );
}

/// One full all-to-all broadcast across n parties.
///
/// Each of the n parties is its own thread. The protocol per thread:
///   1. sleep(latency)                                <-- wire propagation
///   2. for each of the n-1 peers:
///        sleep(transmission_per_msg)                 <-- bandwidth-limited uplink
///        senders[peer].send(payload)
///   3. recv() n-1 messages from own inbox
fn run_broadcast(
    n: usize,
    payloads: &[Vec<u8>],
    latency: Duration,
    transmission_per_msg: Duration,
) {
    let mut senders = Vec::with_capacity(n);
    let mut inboxes = Vec::with_capacity(n);
    for _ in 0..n {
        let (tx, rx) = mpsc::channel::<Vec<u8>>();
        senders.push(tx);
        inboxes.push(rx);
    }

    let mut handles = Vec::with_capacity(n);
    for (i, inbox) in inboxes.into_iter().enumerate() {
        let senders_clone: Vec<_> = senders.iter().cloned().collect();
        let my_payload = payloads[i].clone();
        handles.push(thread::spawn(move || {
            // --- (1) wire propagation latency ---
            thread::sleep(latency);

            // --- (2) bandwidth-limited broadcast ---
            for j in 0..n {
                if j != i {
                    if !transmission_per_msg.is_zero() {
                        thread::sleep(transmission_per_msg);
                    }
                    senders_clone[j].send(my_payload.clone()).unwrap();
                }
            }

            // --- (3) receive n-1 messages ---
            for _ in 0..(n - 1) {
                let _ = inbox.recv().unwrap();
            }
        }));
    }

    drop(senders);
    for h in handles {
        let _ = h.join();
    }
}

/// Time to push one message of `payload_bytes` bytes onto an uplink of
/// `bandwidth_mbps` Mbps. Returns Duration::ZERO if bandwidth_mbps == 0
/// (infinite bandwidth: latency-only model).
fn transmission_per_msg(payload_bytes: usize, bandwidth_mbps: u64) -> Duration {
    if bandwidth_mbps == 0 {
        return Duration::ZERO;
    }
    let bytes_per_sec = bandwidth_mbps.saturating_mul(1_000_000) / 8;
    Duration::from_secs_f64(payload_bytes as f64 / bytes_per_sec as f64)
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

fn format_bytes(b: usize) -> String {
    if b >= 1_048_576 {
        format!("{:.2} MB", b as f64 / 1_048_576.0)
    } else if b >= 1024 {
        format!("{:.2} KB", b as f64 / 1024.0)
    } else {
        format!("{b} B")
    }
}