# Ladon - Semi-Honest (Passive) Variant

This crate implements the **semi-honest variant** of Ladon: honest-majority
threshold KEM (`n = 2t + 1`) over a prime modulus, using Shamir secret sharing
and the negacyclic NTT for polynomial arithmetic.

## Underlying ML-KEM library

The underlying functionality is built on the very efficient Rust
implementation of the ML-KEM: [KEMKEM](https://github.com/conorpo/kemkem).

We retarget it to a 30-bit NTT-friendly prime modulus `q = 687659009` with
primitive root `ZETA = 174453379` and adopt Ladon's parameter sets
(`Ladon128`, `Ladon256` - see `params.rs`). All threshold and MPC machinery
(Shamir sharing, dealer, party operations, double sharings, `receiver_reconstruct`)
is implemented from scratch in this crate.

## Build and run

Requires **nightly Rust (1.94 at least)** (uses `feature(generic_const_exprs)`):

```bash
rustup default nightly
```

### Demos

```bash
cargo run --release --bin demo_kem
cargo run --release --bin demo_threshold_kem
```