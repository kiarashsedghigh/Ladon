"""
Compression Error Analysis for Lattice-Based Cryptography.

This module analyzes compression-decompression errors in MLWE-based public-key encryption schemes.
Uses exact rational arithmetic via the prob_dist (probability distribution) module for precise error_term_analyzer distribution
computation.

Key Features:
- Exact error_term_analyzer distributions using Decimal arithmetic
- Parallel processing support for large moduli
- Unified compression function for any modulus reduction
- Compatible with MLWE-based PKE compression (use p = 2^d for d-bit compression)
- Detailed statistical analysis
"""

from __future__ import annotations
from typing import Dict, Optional
from decimal import Decimal
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import multiprocessing

from error_term_analyzer.probability_distribution import *


# =============================================================================
# Core Compression Function
# =============================================================================

def _modulus_reduction_error(x: int, q: int, p: int) -> int:
    """
    Compute the compression-decompression error_term_analyzer for modulus reduction.

    Compresses x from Z_q to Z_p and back, computing the error_term_analyzer:
    Error = decompress(compress(x)) - x, centered in [-q/2, q/2).

    Compression:   y = ?(p/q) * x? mod p
    Decompression: x' = ?(q/p) * y?
    Error:         e = x' - x (centered)

    Args:
        x: Original value in Z_q.
        q: Source modulus (typically prime, e.g., 3329 in many MLWE-based schemes).
        p: Target modulus (typically 2^d for d-bit compression).

    Returns:
        Error value, centered in [-q/2, q/2).

    Examples:
        >>> # 4-bit compression of a ciphertext component (d=4, so p=2^4=16)
        >>> _modulus_reduction_error(100, q=3329, p=16)
        -5

        >>> # 10-bit compression of a ciphertext component (d=10, so p=2^10=1024)
        >>> _modulus_reduction_error(100, q=3329, p=1024)
        0

        >>> # General modulus reduction
        >>> _modulus_reduction_error(1000, q=3329, p=256)
        -14

    Note:
        - For d-bit compression, set p = 2^d
        - The error_term_analyzer distribution depends on both q and p
        - Centering ensures error_term_analyzer is in [-q/2, q/2) for proper analysis
    """
    # Compress: Z_q -> Z_p
    compressed = round((p * x) / q) % p

    # Decompress: Z_p -> Z_q
    decompressed = round((q * compressed) / p)

    # Compute error_term_analyzer
    error = decompressed - x

    # Center the error_term_analyzer in [-q/2, q/2)
    if error >= q // 2:
        error -= q
    elif error < -(q // 2):
        error += q

    return error


# =============================================================================
# Error Distribution Computation
# =============================================================================

def _count_modulus_reduction_errors_batch(x_values: range, q: int, p: int) -> Dict[int, int]:
    """
    Compute compression errors for a batch of x values.

    Args:
        x_values: Range of x values to process.
        q: Source modulus.
        p: Target modulus.

    Returns:
        Dictionary mapping error_term_analyzer values to their counts.

    Note:
        Internal function used for parallel processing.
    """
    error_counts = defaultdict(int)
    for x in x_values:
        error = _modulus_reduction_error(x, q, p)
        error_counts[error] += 1
    return dict(error_counts)


def _merge_error_counts(count_dicts: list[Dict[int, int]]) -> Dict[int, int]:
    """
    Merge multiple error_term_analyzer count dictionaries.

    Args:
        count_dicts: List of dictionaries mapping errors to counts.

    Returns:
        Merged dictionary.

    Note:
        Internal function used for parallel processing.
    """
    merged = defaultdict(int)
    for count_dict in count_dicts:
        for error, count in count_dict.items():
            merged[error] += count
    return dict(merged)


def compute_modulus_reduction_error_distribution(
        q: int,
        p: int,
        use_parallel: Optional[bool] = None,
        num_workers: Optional[int] = None
) -> ProbabilityDistribution:
    """
    Compute the exact distribution of compression-decompression errors.

    For a uniformly random x from Z_q, compute the distribution of
    the error_term_analyzer when compressing to Z_p and back, using exact Decimal arithmetic.

    Args:
        q: Source modulus (size of Z_q).
        p: Target modulus (typically 2^d for d-bit compression).
        use_parallel: Whether to use parallel processing.
                     If None, auto-decide based on q (parallel if q > 10000).
        num_workers: Number of worker processes (default: CPU count).

    Returns:
        ProbabilityDistribution of compression errors with exact probabilities.

    Examples:
        >>> # Parameters for a 4-bit compression error distribution
        >>> e_prime_dist = compute_modulus_reduction_error_distribution(q=3329, p=2**4)
        >>> e_prime_dist.mean()
        Decimal(0/ 1)

        >>> # Parameters for a 10-bit compression error distribution
        >>> e_double_prime_dist = compute_modulus_reduction_error_distribution(q=3329, p=2**10)
        >>> e_double_prime_dist.support_size
        3

    Note:
        - Returns exact Decimal probabilities (not floats)
        - Automatically uses parallel processing for large q
        - Computation is O(q), so can be slow for very large moduli
    """
    # Auto-decide whether to use parallel processing
    if use_parallel is None:
        use_parallel = q > 10000

    if not use_parallel or q < 1000:
        # Single-threaded version for small q
        error_counts = defaultdict(int)
        for x in range(q):
            error = _modulus_reduction_error(x, q, p)
            error_counts[error] += 1
    else:
        # Parallel version for large q
        if num_workers is None:
            num_workers = multiprocessing.cpu_count()

        # Split work into chunks (4 chunks per worker for load balancing)
        chunk_size = max(1, q // (num_workers * 4))
        x_ranges = []
        for i in range(0, q, chunk_size):
            x_ranges.append(range(i, min(i + chunk_size, q)))

        # Use ProcessPoolExecutor for CPU-bound work
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Create partial function with fixed q and p
            compute_batch = partial(_count_modulus_reduction_errors_batch, q=q, p=p)

            # Map the work across processes
            count_dicts = list(executor.map(compute_batch, x_ranges))

        # Merge all results
        error_counts = _merge_error_counts(count_dicts)

    # Convert counts to exact probabilities using Decimal
    error_values = sorted(error_counts.keys())
    probabilities = [Decimal(error_counts[e] / q) for e in error_values]

    return ProbabilityDistribution(error_values, probabilities)
