![Logo](logo.png)
<h1 align="center">Ladon - Decentralizing Key Management Service via a Threshold KEM</h1>

<p align="center">
  <a href="#overview">Overview</a> ·
  <a href="#repository-structure">Structure</a> ·
  <a href="#reviewer-instructions">Reviewer Instructions</a> ·
</p>

---

## Overview

**Ladon** is a post-quantum secure threshold Key Encapsulation Mechanism (KEM) for confidential-computing deployments. It replaces the centralized Key Management Service (KMS) in the key-release pipeline with a committee of KMS authorities, eliminating the single point of compromise inherent to existing KMS architectures while preserving the IND-CCA security guarantees of NIST-standardized ML-KEM.

Ladon builds on the structural separation in ML-KEM between (i) a linear, secret-key-dependent decryption step and (ii) the symmetric Fujisaki-Okamoto (FO) finalization. Only the linear step is distributed across the KMS committee through a secret-sharing-based threshold protocol; the FO finalization runs locally inside the trusted execution environment (TEE), where the recovered key already needs to land in the clear. This partition keeps Ladon functionally equivalent to single-party ML-KEM while avoiding the heavy multi-party machinery (e.g., garbled circuits) that prior threshold KEM constructions require.

This repository provides a Rust implementation of Ladon covering **both** the semi-honest and active-security variants:

- **Semi-honest variant** — honest-majority setting (`n = 2t + 1`), using Shamir secret sharing over a prime field `Z_q`. Ring multiplication is accelerated via the negacyclic NTT, as the modulus is an NTT-friendly prime.
- **Active variant** — dishonest-majority setting (`t = n - 1`), using SPDZ₂ₖ authenticated additive secret sharing over `Z_{2^k}` with information-theoretic MACs. The modulus is a power of two, so the NTT does not apply; ring multiplication is implemented as a dense negacyclic matrix-vector product using Rust's native `i128` type and the `ndarray` crate.

In both variants, the threshold and MPC machinery (Shamir reconstruction, SPDZ₂ₖ MAC authentication, double-sharing generation, BatchMACCheck) is implemented from scratch and integrated directly with the lattice arithmetic of `R_q`. The only external cryptographic dependency is an existing ML-KEM library, from which we reuse key generation, encapsulation, and the FO finalization step, so that the core PKE primitive remains the NIST-standardized construction.

## Repository Structure

```
ladon/
├── semi-honest/   Rust implementation of the semi-honest (passive)
│                  variant over the prime modulus q = 687659009,
│                  with Shamir secret sharing in the honest-majority
│                  setting.
│
├── active/        Rust implementation of the active (malicious-secure)
│                  variant over the power-of-two modulus q = 2^30, with
│                  SPDZ₂ₖ authenticated additive sharing in the
│                  dishonest-majority setting.
│
├── param_analyze/  Set of scripts for analyzing the parameter 
│                   selection of Ladon, including its underlying
│                   KEM's security, noise analysis, and threshold
│                   performance (e.g., sucess aimplification).
```

Each variant is a self-contained Rust crate with its own `Cargo.toml`, source tree, demos, and benchmarks. They share the same high-level protocol structure (KMS Committee Provision, Owner Asset Key Wrapping, KMS Distributed Decryption, TEE Asset Key Derivation) and the same parameter sets (Ladon128 and Ladon256), but differ in the underlying secret-sharing scheme and ring arithmetic as described above.

## Reviewer Instructions

To run each instance:

1. **Semi-honest variant.**
   ```bash
   cd semi-honest
   ```
   Then follow the instructions in `semi-honest/README.md`.

2. **Active variant.**
   ```bash
   cd active
   ```
   Then follow the instructions in `active/README.md`.

The two crates are independent; they can be built and benchmarked in either order.

For the scheme analysis, see `param_analyze/README.md`.
