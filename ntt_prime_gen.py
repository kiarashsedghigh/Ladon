"""
Generate a prime q suitable for NTT over Z_q[x] / (x^N + 1), plus a primitive
2N-th root of unity psi (so psi^N == -1 mod q) and the corresponding primitive
N-th root omega = psi^2.

By default the script returns the *smallest* primitive 2N-th root of unity for
the chosen q (deterministic, reproducible). Pass -f / --fast to use a random
search instead (much faster for very large q, but the returned psi is random).

Usage:
    python ntt_prime.py                 # defaults: N=256, ~23-bit prime, smallest psi
    python ntt_prime.py -N 256 -b 32
    python ntt_prime.py -f              # fast random search for psi
    python ntt_prime.py --smallest      # smallest prime >= 2^(bits-1) instead of random
"""

import argparse
import random
import sys


# -----------------------------------------------------------------------------
# Miller-Rabin (deterministic for the bit ranges we use here, but with k=40
# random witnesses it is overwhelmingly safe for any size we care about).
# -----------------------------------------------------------------------------
def is_prime(n, k=40):
    if n < 2:
        return False
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for p in small_primes:
        if n == p:
            return True
        if n % p == 0:
            return False

    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1

    for _ in range(k):
        a = random.randrange(2, n - 1)
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True


# -----------------------------------------------------------------------------
# Prime search: q = k * 2N + 1
# -----------------------------------------------------------------------------
def find_prime(N, bits, smallest=False):
    two_n = 2 * N
    lo = 1 << (bits - 1)
    hi = (1 << bits) - 1

    # Smallest k such that k*2N + 1 >= lo
    k_lo = (lo - 1 + two_n - 1) // two_n
    k_hi = (hi - 1) // two_n
    if k_lo > k_hi:
        raise ValueError(f"No q = k*{two_n}+1 fits in {bits} bits.")

    if smallest:
        for k in range(k_lo, k_hi + 1):
            q = k * two_n + 1
            if is_prime(q):
                return q
        raise RuntimeError("No prime found in range.")
    else:
        # Randomized search
        for _ in range(1 << 20):
            k = random.randint(k_lo, k_hi)
            q = k * two_n + 1
            if is_prime(q):
                return q
        raise RuntimeError("Prime search exhausted.")


# -----------------------------------------------------------------------------
# Find a primitive 2N-th root of unity in Z_q*.
# For N a power of two, psi has order exactly 2N iff psi^N == -1 mod q
# (orders divide 2N and are themselves powers of two, so any smaller order
# would give psi^N == 1).
#
#   smallest: iterate psi = 2, 3, ... and return the first one with psi^N == -1.
#   fast:     pick random a, set psi = a^((q-1)/(2N)); same condition.
# -----------------------------------------------------------------------------
def find_primitive_2n_root(q, N, fast=False):
    assert (q - 1) % (2 * N) == 0, "Need q ≡ 1 (mod 2N)."
    if fast:
        exp = (q - 1) // (2 * N)
        for _ in range(1 << 16):
            a = random.randrange(2, q - 1)
            psi = pow(a, exp, q)
            if pow(psi, N, q) == q - 1:
                return psi
        raise RuntimeError("Could not find a primitive 2N-th root of unity.")
    # Smallest: deterministic linear scan.
    for psi in range(2, q):
        if pow(psi, N, q) == q - 1:
            return psi
    raise RuntimeError("No primitive 2N-th root of unity exists in Z_q.")


# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-N", type=int, default=256, help="ring dimension (default 256)")
    ap.add_argument("-b", "--bits", type=int, default=30,
                    help="target bit length of q (default 23)")
    ap.add_argument("-f", "--fast", action="store_true",
                    help="random search for psi (default: smallest psi)")
    ap.add_argument("--smallest", action="store_true",
                    help="return the smallest valid prime in the bit range")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    N = args.N
    if N & (N - 1):
        sys.exit("N must be a power of two for NTT over x^N + 1.")

    q     = find_prime(N, args.bits, smallest=args.smallest)
    psi   = find_primitive_2n_root(q, N, fast=args.fast)
    omega = pow(psi, 2, q)
    # Modular inverses, often handy for the inverse NTT
    psi_inv   = pow(psi, -1, q)
    omega_inv = pow(omega, -1, q)
    N_inv     = pow(N, -1, q)

    # Sanity
    assert pow(psi, N, q) == q - 1
    assert pow(psi, 2 * N, q) == 1
    assert pow(omega, N, q) == 1
    assert pow(omega, N // 2, q) == q - 1

    print(f"N         = {N}")
    print(f"q         = {q}   ({q.bit_length()} bits)")
    print(f"q-1       = {q - 1} = {(q - 1) // (2 * N)} * 2N")
    print(f"psi       = {psi}        # primitive 2N-th root, psi^N = -1 mod q")
    print(f"psi^-1    = {psi_inv}")
    print(f"omega     = {omega}      # primitive N-th root, = psi^2")
    print(f"omega^-1  = {omega_inv}")
    print(f"N^-1      = {N_inv}")


if __name__ == "__main__":
    main()