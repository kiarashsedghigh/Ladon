# Moiragus

Analysis toolkit for MLWE-based PKE schemes. Covers parameter selection, decryption error estimation, compression error bounds, and security analysis against known lattice attacks.

---

## Project Structure

```
Moiragus/
├── moiragus_base_pke.py             # Main script — full scheme analysis (requires SageMath)
├── moiragus_threshold_repeatition.py# Main script — majority-vote failure analysis (Python 3)
├── parameter_selection/             # Parameter set definition and key/ciphertext size computation
├── error_term_analyzer/             # Error term and compression error analysis
└── security_analysis/
    ├── mlwe_pke_security.py         # Security level estimation via LWE reduction
    └── estimator/                   # Lattice Estimator (bundled, no install needed)
```

---

## Tools

| Module | Description |
|---|---|
| `parameter_selection` | Defines `MLWEPKEParams` and computes secret key, public key, and ciphertext sizes |
| `error_term_analyzer` | Computes the full decryption error distribution for the error term rᵀe + e₂ + e′ − (e₁ + e″)ᵀs, including compression errors from u and v |
| `security_analysis` | Reduces the scheme to an equivalent LWE instance and estimates bit security against primal, dual, and BKZ attacks |

The [Lattice Estimator](https://github.com/malb/lattice-estimator) is already bundled under `security_analysis/estimator/` — no separate installation is required.

---

## Running the Scripts

### 1. `moiragus_base_pke.py` — Full Scheme Analysis

Computes key/ciphertext sizes, decryption error probability, and security level via the Lattice Estimator. **Requires SageMath** because the Lattice Estimator depends on Sage internals.

```bash
sage -python moiragus_base_pke.py
```

Edit the `MLWEPKEParams` block at the top of the script to configure the parameter set before running.

### 2. `moiragus_threshold_repeatition.py` — Majority-Vote Failure Analysis

Given a single-execution decryption failure probability `pe = 2^bits` (entered at runtime) and a repetition count `k`, computes the probability that a majority vote over `k` independent decryptions still fails. **Runs with standard Python 3**, no extra dependencies.

```bash
python3 moiragus_threshold_repeatition.py
```

The script will prompt for the `pe` exponent (e.g. `-9` for `pe = 2⁻⁹`).

---

## Requirements

- [SageMath](https://www.sagemath.org/) — required for `moiragus_base_pke.py` only
- Python 3.8+ — sufficient for `moiragus_threshold_repeatition.py`
