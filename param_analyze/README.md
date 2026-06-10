# Ladon - Parameter Analysis

Tooling for selecting and validating the MLWE parameters used by the **Ladon** threshold KEM (passive and active variants). The entry point is `base_pke.py`; everything else is a supporting module.

## Table of Contents

- [Requirements](#requirements)
- [Running](#running)
- [Parameter struct](#parameter-struct)
- [Crypto-size computation](#crypto-size-computation)
- [Analysis pipeline](#analysis-pipeline)
  - [1. MLWE security (lattice estimator)](#1-mlwe-security-lattice-estimator)
  - [2. Decryption-error / noise analysis](#2-decryption-error--noise-analysis)
  - [3. Success amplification](#3-success-amplification)
- [Communication cost](#communication-cost)

## Requirements

The MLWE security estimator depends on [SageMath](https://www.sagemath.org/) — `security_analysis/` imports the [lattice-estimator](https://github.com/malb/lattice-estimator), which only runs under Sage's Python. Install Sage system-wide (e.g. `apt install sagemath` on Ubuntu, or via conda-forge) and make sure `sage` is on your `PATH`.

## Running

Run the entry point through Sage's bundled Python:

```bash
sage -python base_pke.py
```

The script is interactive: it walks each parameter set in turn and prompts for the error bound, `t` (passive), `ℓ`, and `λ_s` (active) so different points in the design space can be explored without editing the file.

## Parameter struct

All analysis is driven by a single immutable dataclass, `MLWEPKEParams` (in `parameter_selection/param_set.py`):

| Field | Meaning |
|-------|---------|
| `q`        | Ring modulus for `R_q = Z_q[X]/(X^d + 1)`. Prime for the passive variant; power of two for the active variant. |
| `k`        | Module rank (number of polynomials). |
| `eta1`     | CBD parameter for key-generation noise (`s`, `e`). |
| `eta2`     | CBD parameter for encryption noise (`e_1`, `e_2`). |
| `du`, `dv` | Compression widths for ciphertext components `u` and `v`. |
| `d`        | Polynomial degree (256 throughout). |

`base_pke.py` instantiates four reference sets — passive 128/256-bit and active 128/256-bit — and feeds them through `analyze_passive` / `analyze_active`.

## Crypto-size computation

Before any analysis, `base_pke.py` computes the byte sizes of every object the protocol moves around:

- `compute_passive_scheme_crypto_sizes` returns `|sk|`, `|sk_share|`, `|pk|`, `|c|`. Shares live in the same ring `Z_q` as the secret.
- `compute_active_scheme_crypto_sizes` additionally accounts for the SPDZ₂ₖ lift: shares and MAC shares live in `Z_{2^{k+λ_s}}`, so `|sk_share|` carries an extra `λ_s` bits per coefficient. It also reports `|sk|` over the original CBD support and over the lifted modulus.

## Analysis pipeline

For each parameter set the script runs three independent checks:

### 1. MLWE security (lattice estimator)

`compute_mlwe_pke_security_level` (in `security_analysis/mlwe_pke_security.py`) builds the equivalent LWE instance — dimension `k·d`, modulus `q`, centered-binomial secret and error with parameter `eta1` — and calls `LWE.estimate.rough` from the [lattice-estimator](https://github.com/malb/lattice-estimator). The output is the rough bit security across the standard attack models (primal-uSVP, dual, etc.).

### 2. Decryption-error / noise analysis

`compute_mlwe_pke_decryption_error` (in `error_term_analyzer/`) builds the exact distribution of the ML-KEM error term

```
r^T e + e_2 + e' − e_1^T s − e''^T s
```

using exact-rational convolutions on `Decimal` probabilities. Compression errors `e'`, `e''` are computed coefficient-by-coefficient from `compute_modulus_reduction_error_distribution` (parallelised for large `q`). The script then applies a union bound over the `d` coefficients and reports the per-coefficient and per-ciphertext failure probability in bits, prompting interactively for the error bound. For the passive scheme it uses the standard `q/4 − 1` bound; for the active scheme it also derives the threshold-failure exponent `N·|e|/μ`.

### 3. Success amplification

The per-decapsulation failure exponent is passed to `run_amplification` (from `success_amplifier`), which reports how many repetitions / how large an `ℓ` are needed to reach the target overall correctness — this is what feeds the BCHK+ ℓ-fold repetition used by Ladon.

## Communication cost

Once `ℓ` is fixed, `cost_scheme_passive` / `cost_scheme_active` (in `communication_cost.py`) print the online-phase bandwidth per party and across the system, broken down by step:

- **Passive:** broadcast masked errors mod μ′, send partial decryptions to the TEE mod q′.
- **Active (SPDZ₂ₖ):** broadcast masked errors mod q/2, broadcast one lifted coefficient for `BatchMACCheck`, send partials to the TEE mod q.

Phase 0 (TEE distributing the ciphertext to the committee) is reported separately so per-party totals can be read with or without it.