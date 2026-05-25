"""
ML-KEM security level estimation.

This module estimates the security level of ML-KEM (Module-Lattice-Based
Key-Encapsulation Mechanism) by constructing equivalent LWE instances and
running lattice attack estimators against them.

References:
    FIPS 203: Module-Lattice-Based Key-Encapsulation Mechanism Standard
    https://github.com/malb/lattice-estimator
"""

from __future__ import annotations

from scheme_analysis.param_set import *
from .estimator import *
from .estimator.lwe_parameters import LWEParameters


def compute_mlkem_security_level(mlkem_params: MLKEMParams) -> dict:
    """
    Estimate the security level of ML-KEM against known lattice attacks.

    Constructs an equivalent LWE (Learning With Errors) instance based on the
    ML-KEM parameters and runs the lattice estimator to assess security against
    known attacks such as BKZ, Dual, and Primal attacks.

    The ML-KEM key encapsulation scheme can be reduced to an LWE problem where:
    - The dimension n equals the total number of ring coefficients (k * d)
    - The modulus q is inherited from the ML-KEM parameters
    - Both the secret and error_term_analyzer distributions are centered binomial with parameter eta1
    - The number of equations m equals the dimension n (square LWE system)

    Args:
        mlkem_params: ML-KEM parameters containing:
            - k: Module dimension (number of polynomials)
            - d: Polynomial degree (coefficients per polynomial)
            - q: The modulus of the ring R_q = Z_q[X]/(X^d + 1)
            - eta1: The standard deviation parameter for the centered binomial
                   distribution used in key generation

    Returns:
        Dictionary containing security estimates with keys corresponding to
        different lattice attack models (e.g., 'bkz', 'dual', 'primal', etc.)
        and their estimated bit security levels.

    Raises:
        ValueError: If ML-KEM parameters are invalid
        RuntimeError: If the estimator cannot complete the security analysis

    Examples:
        >>> from scheme_analysis.param_set import MLKEM512_PARAMS
        >>> security_estimates = compute_mlkem_security_level(MLKEM512_PARAMS)
        >>> print(f"Estimated security: {security_estimates}")

        >>> # For a specific parameter set
        >>> from scheme_analysis.param_set import MLKEM768_PARAMS
        >>> estimates = compute_mlkem_security_level(MLKEM768_PARAMS)
        >>> print(f"ML-KEM-768 security level: {estimates}")

    Notes:
        - The security level is estimated against known attacks in the lattice-based
          cryptanalysis literature.
        - Results depend on the lattice estimator version and its internal models.
        - This is a rough/conservative estimate; actual security may vary based on
          optimized attack implementations.
    """
    print("=" * 60)
    print("ML-KEM MLWE Security Estimation")
    print("=" * 60)
    # Construct equivalent LWE instance from ML-KEM parameters
    # The dimension n = total number of ring coefficients
    # Each polynomial has degree d, and there are k polynomials in the module
    lwe_instance = LWEParameters(
        n=mlkem_params.k * mlkem_params.d,  # Total dimension: module dimension * polynomial degree
        q=mlkem_params.q,  # Modulus inherited from ML-KEM
        Xs=ND.CenteredBinomial(mlkem_params.eta1),  # Secret distribution (centered binomial)
        Xe=ND.CenteredBinomial(mlkem_params.eta1),  # Error distribution (centered binomial)
        m=mlkem_params.k * mlkem_params.d,  # Number of equations (square LWE system)
    )

    # Run security estimation against known lattice attacks
    # This performs a rough analysis of the security level
    return LWE.estimate.rough(lwe_instance)