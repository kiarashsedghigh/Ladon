"""
Majority Vote Failure Probability Calculator

Given a single-decryption failure probability p_dec_fail and k parallel
decryptions combined by coefficient-wise majority, the total failure
probability is bounded by:

    p_total_fail(k) = sum_{i=ceil(k/2)}^{k} C(k,i) * p_dec_fail^i * (1 - p_dec_fail)^(k-i)
"""
from math import comb, log2, ceil


def p_total_fail(p_dec_fail: float, k: int) -> float:
    """Probability that at least ceil(k/2) of k independent trials fail."""
    threshold = ceil(k / 2)
    return sum(
        comb(k, i) * p_dec_fail**i * (1 - p_dec_fail) ** (k - i)
        for i in range(threshold, k + 1)
    )


def run_amplification(pe_exponent = -8) -> None:
    # ---- parameters ----
    # pe_exponent = -8           # p_dec_fail = 2^pe_exponent
    k_min, k_max = 1, 51
    # --------------------

    p_dec_fail = 2.0 ** pe_exponent
    print(f"p_dec_fail = 2^{pe_exponent} = {p_dec_fail:.6e}")
    print(f"k = {k_min}..{k_max}\n")

    header = f"{'k':>4} | {'p_total_fail':>14} | {'log2':>8} | {'1 - p_total_fail':>20}"
    print(header)
    print("-" * len(header))

    for k in range(k_min, k_max + 1):
        pf = p_total_fail(p_dec_fail, k)
        log_str = f"{log2(pf):>8.3f}" if pf > 0 else "    -inf"
        print(f"{k:>4} | {pf:>14.6e} | {log_str} | {1 - pf:>20.15f}")

if __name__ == "__main__":
    run_amplification()