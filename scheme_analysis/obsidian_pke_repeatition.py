"""
Majority Vote Failure Probability Calculator

Given:
    pe : probability that a single decryption fails
    k  : number of independent decryptions

The probability that the majority vote FAILS (more than half of k results are wrong) is:

    p_fail(k) = sum_{x=ceil(k/2)}^{k} C(k, x) * pe^x * (1 - pe)^(k - x)

The probability that the majority vote is CORRECT is:
    pc(k) = 1 - p_fail(k)

This script lists pc (and the failure probability) for a range of odd k values,
given pe set in the PARAMETERS section below.
"""

from math import comb, log2, ceil

# ============================================================
# PARAMETERS (edit these — no command-line arguments)
# ============================================================

# Single-execution failure probability, expressed as 2^(PE_EXPONENT).
# Example: PE_EXPONENT = -10 means pe = 2^-10 ≈ 0.0009765625
PE_EXPONENT = -10

# Range of k (number of independent decryptions) to evaluate.
# Only odd k values are used so that "more than half" is unambiguous.
K_MIN = 1
K_MAX = 51
ONLY_ODD_K = False  # set to False to include even k as well

# ============================================================


def majority_failure_probability(pe: float, k: int) -> float:
    """
    Probability that more than half of k independent trials fail,
    each with individual failure probability pe.
    """
    threshold = ceil(k / 2 + 1e-12)  # strict majority: x > k/2
    # For odd k, ceil(k/2) already gives (k+1)/2 which is the strict majority.
    # For even k, "more than half" means x >= k/2 + 1.
    if k % 2 == 0:
        threshold = k // 2 + 1
    else:
        threshold = (k + 1) // 2

    total = 0.0
    for x in range(threshold, k + 1):
        total += comb(k, x) * (pe ** x) * ((1 - pe) ** (k - x))
    return total


def safe_log2(x: float) -> str:
    """Format log2(x) nicely, handling x = 0 or extremely small x."""
    if x <= 0:
        return "  -inf"
    try:
        return f"{log2(x):>8.3f}"
    except ValueError:
        return "  -inf"


def main() -> None:
    pe = 2.0 ** PE_EXPONENT
    print(f"Single-execution failure probability: pe = 2^{PE_EXPONENT} = {pe:.6e}")
    print(f"Evaluating k from {K_MIN} to {K_MAX}"
          f"{' (odd values only)' if ONLY_ODD_K else ''}\n")

    header = f"{'k':>4} | {'p_fail':>14} | {'log2(p_fail)':>12} | {'pc = 1 - p_fail':>20}"
    print(header)
    print("-" * len(header))

    for k in range(K_MIN, K_MAX + 1):
        if ONLY_ODD_K and k % 2 == 0:
            continue
        p_fail = majority_failure_probability(pe, k)
        pc = 1.0 - p_fail
        print(f"{k:>4} | {p_fail:>14.6e} | {safe_log2(p_fail)} | {pc:>20.15f}")

    # Sanity check matching the paper's example:
    # pe ≈ 2^-9, k = 7  ->  p_fail ≈ 2^-31
    print("\nSanity check from the paper (pe = 2^-9, k = 7):")
    pe_check = 2.0 ** -9
    p_fail_check = majority_failure_probability(pe_check, 7)
    print(f"  p_fail = {p_fail_check:.6e}  ->  log2(p_fail) ≈ {log2(p_fail_check):.3f}")
    print(f"  (paper claims approximately 2^-31)")


if __name__ == "__main__":
    main()