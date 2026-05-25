"""
Probability Distribution Module for Cryptographic Error Analysis.

This module provides high-precision decimal arithmetic for probability distributions
used in analyzing decryption errors in lattice-based cryptography (LWE, Ring-LWE, Module-LWE).

Key Features:
- High-precision arithmetic using Python's Decimal class (configurable precision)
- Efficient convolution operations using NumPy
- Support for binomial and uniform distributions
- Distribution operations (multiplication, exponentiation, product distributions)
- Better performance than Fraction-based implementation
"""

from __future__ import annotations
from typing import Dict, List, Union, Iterable, Optional, Literal
from decimal import Decimal, getcontext
import numpy as np
import multiprocessing as mp

# Set high precision for cryptographic applications
# 100 decimal places provides extreme accuracy while maintaining performance
getcontext().prec = 100

__version__ = "2.0.0"
__all__ = [
    "ProbabilityDistribution",
    "sample_binomial_distribution",
    "sample_uniform_distribution",
    "multiply_probability_distributions"
]

EPSILON = Decimal(10) ** -(getcontext().prec - 10)


# Worker function (must be at module level for pickling)
def _compute_power_worker(dist: ProbabilityDistribution, n: int) -> ProbabilityDistribution:
    """
    Worker function for parallel computation.
    Computes dist^n using binary exponentiation.
    """
    return dist._binary_exponentiation(n)


class ProbabilityDistribution:
    """
    A discrete probability distribution with high-precision decimal arithmetic.

    Represents a probability distribution as a sparse dictionary mapping
    values to their probabilities (as Decimal objects). Supports
    efficient convolution operations for computing sums of independent
    random variables.

    Attributes:
        dist (Dict[int, Decimal]): Sparse representation mapping values to probabilities.

    Examples:
        >>> # Create a simple distribution
        >>> dist = ProbabilityDistribution([0, 1, 2], [0.25, 0.5, 0.25])

        >>> # Compute sum of two independent copies
        >>> sum_dist = dist * dist

        >>> # Compute sum of n independent copies
        >>> n_sum = dist ** 10
    """

    def __init__(
            self,
            values: Iterable[int],
            probabilities: Iterable[Union[Decimal, int, float, str]]
    ) -> None:
        """
        Initialize a probability distribution.

        Args:
            values: Possible values the random variable can take.
            probabilities: Corresponding probabilities (will be normalized and converted to Decimal).

        Raises:
            AssertionError: If values and probabilities have different lengths.
            AssertionError: If probabilities don't sum to 1 after normalization.

        Note:
            Probabilities are automatically normalized to sum to exactly 1.
            Zero probabilities are automatically filtered out for efficiency.
        """
        values = list(values)
        probabilities = list(probabilities)

        assert len(values) == len(probabilities), (
            f"Values and probabilities must have same length: "
            f"got {len(values)} values and {len(probabilities)} probabilities"
        )

        # Convert all probabilities to Decimal
        probabilities = [
            p if isinstance(p, Decimal) else Decimal(str(p))
            for p in probabilities
        ]

        # Normalize to ensure sum of 1
        prob_sum = sum(probabilities)
        if abs(prob_sum - Decimal(1)) > EPSILON:
            probabilities = [p / prob_sum for p in probabilities]

        # Verify sum is (approximately) 1
        total = sum(probabilities)
        assert abs(total - Decimal(1)) < EPSILON, f"Probabilities must sum to 1, got {total}"

        # Store as sparse dictionary: value -> probability
        self.dist: Dict[int, Decimal] = {}
        for v, p in zip(values, probabilities):
            if abs(p) > EPSILON:  # Only store non-zero probabilities
                self.dist[v] = p

    @property
    def support(self) -> List[int]:
        """Get the support (all values with non-zero probability)."""
        return sorted(self.dist.keys())

    @property
    def support_size(self) -> int:
        """Get the size of the support."""
        return len(self.dist)

    def get_coefficients(self) -> Dict[int, Decimal]:
        """
        Get the coefficient dictionary.

        Returns:
            Dictionary mapping values to their probabilities.
        """
        return self.dist.copy()

    def probability(self, value: int) -> Decimal:
        """
        Get the probability of a specific value.

        Args:
            value: The value to query.

        Returns:
            The probability (0 if value not in support).
        """
        return self.dist.get(value, Decimal(0))

    def probability_in_range(self, alpha: int) -> Decimal:
        """
        Compute P[|X| <= alpha].

        Args:
            alpha: Range bound (inclusive).

        Returns:
            Probability that the absolute value is at most alpha.
        """
        prob = Decimal(0)
        for val, p in self.dist.items():
            if abs(val) <= alpha:
                prob += p
        return prob

    def mean(self) -> Decimal | Literal[0]:
        """
        Compute the mean (expected value).

        Returns:
            E[X] as a Decimal.
        """
        return sum(Decimal(v) * p for v, p in self.dist.items())

    def variance(self) -> Decimal:
        """
        Compute the variance.

        Returns:
            Var[X] = E[X^2] - E[X]^2 as a Decimal.
        """
        mean_val = self.mean()
        second_moment = sum(Decimal(v * v) * p for v, p in self.dist.items())
        return second_moment - mean_val * mean_val

    def _to_dense_array(self) -> tuple[np.ndarray, int]:
        """
        Convert sparse distribution to dense NumPy array.

        Returns:
            Tuple of (array, offset) where array contains Decimal objects
            and offset is the value corresponding to index 0.
        """
        if not self.dist:
            return np.array([Decimal(1)], dtype=object), 0

        min_val = min(self.dist.keys())
        max_val = max(self.dist.keys())

        # Create dense array with object dtype for Decimals
        size = max_val - min_val + 1
        array = np.zeros(size, dtype=object)

        # Initialize with Decimal(0)
        array[:] = Decimal(0)

        for val, prob in self.dist.items():
            array[val - min_val] = prob

        return array, min_val

    @staticmethod
    def _from_dense_array(array: np.ndarray, offset: int) -> ProbabilityDistribution:
        """
        Convert dense NumPy array to sparse ProbabilityDistribution.

        Args:
            array: NumPy array of Decimal probabilities.
            offset: Value corresponding to index 0.

        Returns:
            A new ProbabilityDistribution object.
        """
        # Find non-zero entries (with tolerance)
        nonzero_indices = [i for i, val in enumerate(array) if abs(val) > EPSILON]

        if len(nonzero_indices) == 0:
            return ProbabilityDistribution([0], [Decimal(1)])

        values = [int(idx + offset) for idx in nonzero_indices]
        probabilities = [array[idx] for idx in nonzero_indices]

        # Renormalize (ensures correctness)
        total = sum(probabilities)
        if abs(total - Decimal(1)) > EPSILON:
            probabilities = [p / total for p in probabilities]

        return ProbabilityDistribution(values, probabilities)

    def __mul__(self, other: ProbabilityDistribution) -> ProbabilityDistribution:
        """
        Convolve two distributions (compute distribution of X + Y).

        For independent random variables X and Y, this computes the
        distribution of their sum using polynomial multiplication.

        Args:
            other: Another ProbabilityDistribution.

        Returns:
            Distribution of X + Y.

        Note:
            Uses NumPy's FFT-based convolution for efficiency.
        """
        # Convert to dense arrays
        arr_a, offset_a = self._to_dense_array()
        arr_b, offset_b = other._to_dense_array()

        # Convolve using NumPy (works with object arrays containing Decimals)
        result_arr = np.convolve(arr_a, arr_b)

        # New offset is sum of offsets
        result_offset = offset_a + offset_b

        # Convert back to sparse representation
        return self._from_dense_array(result_arr, result_offset)

    def __pow__(self, n: int) -> ProbabilityDistribution:
        """
        Compute n-fold convolution (distribution of sum of n independent copies).

        For a random variable X, this computes the distribution of
        X_1 + X_2 + ... + X_n where all X_i are independent copies of X.

        Args:
            n: Number of independent copies to sum (non-negative integer).

        Returns:
            Distribution of the sum.

        Note:
            Uses binary exponentiation for O(log n) complexity.
        """
        if n == 0:
            return ProbabilityDistribution([0], [Decimal(1)])
        if n == 1:
            return self

        # Use binary exponentiation (repeated squaring)
        result = ProbabilityDistribution([0], [Decimal(1)])
        base = self

        while n > 0:
            if n % 2 == 1:
                result = result * base
            base = base * base
            n //= 2

        return result

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"ProbabilityDistribution(support_size={len(self.dist)})"

    def __str__(self) -> str:
        """Human-readable string representation."""
        items = sorted(self.dist.items())[:5]  # Show first 5 items
        items_str = ", ".join(f"{v}: {float(p):.6f}" for v, p in items)
        if len(self.dist) > 5:
            items_str += ", ..."
        return f"ProbabilityDistribution({{{items_str}}})"

    def print_distribution(self, name: str = "C") -> None:
        """
        Print the distribution as a polynomial.

        Args:
            name: Name to use in the output (default "C").
        """
        print(f"\n{name}(X) = ", end="")

        terms = []
        for val in sorted(self.dist.keys()):
            prob = self.dist[val]
            if abs(prob) > EPSILON:
                prob_str = f"{float(prob):.6f}"
                if val == 0:
                    terms.append(f"{prob_str}")
                elif val > 0:
                    terms.append(f"{prob_str}·X^{val}")
                else:
                    terms.append(f"{prob_str}·X^({val})")

        print(" + ".join(terms))

    def print_probabilities(self, name: str = "C") -> None:
        """
        Print probabilities as a formatted table.

        Args:
            name: Name to use in the table header (default "C").
        """
        print(f"\nProbabilities for {name}:")
        print(f"{'Value':<10} {'Probability (Decimal)':<50} "
              f"{'Probability (float)':<20} {'Cumulative':<20}")
        print("-" * 100)

        cumulative = Decimal(0)
        for val in sorted(self.dist.keys()):
            prob = self.dist[val]
            cumulative += prob
            print(f"{val:<10} {str(prob):<50} "
                  f"{float(prob):<20.12f} {float(cumulative):<20.12f}")


# =============================================================================
# Distribution Sampling Functions
# =============================================================================

def sample_binomial_distribution(beta: int) -> ProbabilityDistribution:
    """
    Create a centered binomial distribution with high-precision probabilities.

    Samples 2*beta uniform bits and computes (# of 1s) - beta,
    resulting in a symmetric distribution over [-beta, beta].

    Probability mass function:
        P(X = k) = C(2*beta, beta + k) / 2^(2*beta)

    Args:
        beta: Half-width of the distribution (beta >= 0).

    Returns:
        ProbabilityDistribution with support [-beta, beta].

    Examples:
        >>> dist = sample_binomial_distribution(2)
        >>> dist.mean()
        Decimal('0')
        >>> dist.support
        [-2, -1, 0, 1, 2]

    Note:
        Commonly used in lattice-based cryptography (e.g., Kyber, Dilithium).
    """
    n = 2 * beta

    def binomial_coefficient(n: int, k: int) -> int:
        """Compute C(n, k) = n! / (k! * (n-k)!) using multiplicative formula."""
        if k < 0 or k > n:
            return 0

        # Use multiplicative formula for efficiency
        result = 1
        for i in range(min(k, n - k)):
            result = result * (n - i) // (i + 1)
        return result

    # Denominator is 2^(2*beta) - compute as Decimal
    denominator = Decimal(2 ** n)

    # Generate values from -beta to beta
    values = list(range(-beta, beta + 1))

    # Compute probabilities
    probabilities = []
    for val in values:
        k = val + beta  # Map to number of 1s

        if k < 0 or k > n:
            probabilities.append(Decimal(0))
        else:
            # Compute binomial coefficient as integer, then convert to Decimal
            coeff = binomial_coefficient(n, k)
            prob = Decimal(coeff) / denominator
            probabilities.append(prob)

    return ProbabilityDistribution(values, probabilities)


def sample_uniform_distribution(beta: int) -> ProbabilityDistribution:
    """
    Create a uniform distribution with high-precision probabilities.

    Equal probability for each value in [-beta, beta].

    Probability mass function:
        P(X = k) = 1 / (2*beta + 1) for all k in [-beta, beta]

    Args:
        beta: Half-width of the distribution (beta >= 0).

    Returns:
        ProbabilityDistribution with support [-beta, beta].

    Examples:
        >>> dist = sample_uniform_distribution(2)
        >>> dist.mean()
        Decimal('0')
        >>> dist.probability(0)
        Decimal('0.2')
    """
    # Generate values from -beta to beta
    values = list(range(-beta, beta + 1))

    # Each value has equal probability
    num_values = 2 * beta + 1
    uniform_prob = Decimal(1) / Decimal(num_values)

    probabilities = [uniform_prob] * num_values

    return ProbabilityDistribution(values, probabilities)


# =============================================================================
# Distribution Operations
# =============================================================================

def multiply_distributions(
        values_a: Iterable[int],
        probs_a: Iterable[Union[Decimal, float, int, str]],
        values_b: Iterable[int],
        probs_b: Iterable[Union[Decimal, float, int, str]]
) -> ProbabilityDistribution:
    """
    Compute the distribution of X * Y for independent random variables.

    Given independent random variables X and Y with known distributions,
    compute the distribution of their product.

    Args:
        values_a: Possible values for X.
        probs_a: Probabilities for X (will be converted to Decimal).
        values_b: Possible values for Y.
        probs_b: Probabilities for Y (will be converted to Decimal).

    Returns:
        Distribution of X * Y.

    Examples:
        >>> # X uniform on {-1, 0, 1}
        >>> X_vals = [-1, 0, 1]
        >>> X_probs = [1/3, 1/3, 1/3]
        >>> # Y uniform on {0, 1}
        >>> Y_vals = [0, 1]
        >>> Y_probs = [1/2, 1/2]
        >>> # Compute X * Y
        >>> product_dist = multiply_distributions(X_vals, X_probs, Y_vals, Y_probs)

    Note:
        This is different from convolution (which computes X + Y).
        Uses O(|supp(X)| * |supp(Y)|) time and space.
    """
    values_a = list(values_a)
    probs_a = list(probs_a)
    values_b = list(values_b)
    probs_b = list(probs_b)

    # Convert to Decimals
    probs_a = [p if isinstance(p, Decimal) else Decimal(str(p)) for p in probs_a]
    probs_b = [p if isinstance(p, Decimal) else Decimal(str(p)) for p in probs_b]

    # Compute all products and their probabilities
    product_probs: Dict[int, Decimal] = {}

    for val_a, prob_a in zip(values_a, probs_a):
        for val_b, prob_b in zip(values_b, probs_b):
            product = int(val_a) * int(val_b)
            prob = prob_a * prob_b

            if product in product_probs:
                product_probs[product] += prob
            else:
                product_probs[product] = prob

    # Convert to sorted lists
    product_values = sorted(product_probs.keys())
    product_prob_list = [product_probs[v] for v in product_values]

    return ProbabilityDistribution(product_values, product_prob_list)


def multiply_probability_distributions(
        prob_a: ProbabilityDistribution,
        prob_b: ProbabilityDistribution
) -> ProbabilityDistribution:
    """
    Compute the distribution of X * Y for two ProbabilityDistribution objects.

    This is a convenience wrapper around `multiply_distributions` that operates
    directly on ProbabilityDistribution instances instead of requiring separate
    value and probability iterables.

    Given two independent random variables X and Y represented as
    ProbabilityDistribution objects, this function computes the distribution
    of their product X * Y.

    Args:
        prob_a: First probability distribution (X).
        prob_b: Second probability distribution (Y).

    Returns:
        A new ProbabilityDistribution representing the product X * Y.

    Examples:
        >>> X = sample_uniform_distribution(1)   # support: {-1, 0, 1}
        >>> Y = sample_uniform_distribution(2)   # support: {-2, -1, 0, 1, 2}
        >>> product_dist = multiply_probability_distributions(X, Y)
        >>> product_dist.support
        [-2, -1, 0, 1, 2]

    Note:
        This operation computes a product distribution, not a convolution.
        Unlike the `*` operator on ProbabilityDistribution (which computes X + Y),
        this function computes X * Y.

        Internally, this function extracts the sparse dictionary representations
        of both distributions and performs a Cartesian product over their supports,
        resulting in O(|supp(X)| * |supp(Y)|) time complexity.
    """
    prob_a_dict = prob_a.get_coefficients()
    prob_b_dict = prob_b.get_coefficients()

    return multiply_distributions(
        prob_a_dict.keys(),
        prob_a_dict.values(),
        prob_b_dict.keys(),
        prob_b_dict.values()
    )
