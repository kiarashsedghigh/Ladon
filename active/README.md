# Ladon - Active (Malicious-Secure) Variant

This crate implements the **active-security variant** of Ladon: dishonest-majority
threshold KEM (`t = n − 1`) over the power-of-two modulus `q = 2^30`, using
SPDZ₂ₖ authenticated additive secret sharing and dense negacyclic
matrix-vector polynomial multiplication.

## Underlying ML-KEM library

The underlying functionality is built on the very efficient Rust
implementation of the ML-KEM: [KEMKEM](https://github.com/conorpo/kemkem).

We retarget it to a power-of-two modulus `q = 2^30` (see `params.rs`),
which is required by the SPDZ₂ₖ threshold protocol (paper §4.5). Because
the NTT does not apply over a power-of-two modulus, ring multiplication is
implemented as a dense negacyclic matrix-vector product using Rust's
native `i128` type and the `ndarray` crate. All threshold and MPC
machinery (SPDZ₂ₖ sharing with the lift to `Z_{2^(k+s)}`, dealer, party
operations, double sharings, `receiver_reconstruct`) is implemented from
scratch in this crate.

## Build and run

Requires **nightly Rust** (uses `feature(generic_const_exprs)`):

```bash
rustup default nightly
```

### Demos

```bash
cargo run --release --bin demo_kem
cargo run --release --bin demo_threshold_kem
```