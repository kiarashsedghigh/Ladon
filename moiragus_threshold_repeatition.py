"""
Majority Vote Failure Probability Calculator

Given:
    pe : single-execution failure probability (specified as 2^bits)
    k  : number of independent decryptions

The probability that the majority vote fails (more than k/2 results are wrong):
    p_fail(k) = sum_{x = ceil(k/2)+1}^{k} C(k, x) * pe^x * (1 - pe)^(k - x)
"""

from math import comb, log2


def majority_failure_probability(pe: float, k: int) -> float:
    """Probability that more than half of k trials fail."""
    threshold = (k + 1) // 2 if k % 2 == 1 else k // 2 + 1
    return sum(comb(k, x) * pe**x * (1 - pe)**(k - x) for x in range(threshold, k + 1))


def print_table(pe_bits: int, k_min: int = 1, k_max: int = 51, only_odd: bool = False) -> None:
    pe = 2.0 ** pe_bits
    print(f"pe = 2^{pe_bits} = {pe:.6e}\n")
    print(f"{'k':>4} | {'p_fail':>14} | {'log2(p_fail)':>12}")
    print("-" * 40)
    for k in range(k_min, k_max + 1):
        if only_odd and k % 2 == 0:
            continue
        p_fail = majority_failure_probability(pe, k)
        log_p = f"{log2(p_fail):>8.3f}" if p_fail > 0 else "    -inf"
        print(f"{k:>4} | {p_fail:>14.6e} | {log_p}")


if __name__ == "__main__":
    pe_bits = int(input("Enter pe exponent (e.g. -9 for pe = 2^-9): "))
    print_table(pe_bits)