"""
MLWE-based PKE security level estimation.

This module estimates the security level of MLWE-based public-key encryption
parameter sets by constructing equivalent LWE instances and running lattice
attack estimators against them.

References:
    https://github.com/malb/lattice-estimator
"""

from __future__ import annotations

from .estimator import *
from .estimator.lwe_parameters import LWEParameters
from parameter_selection.param_set import MLWEPKEParams

def compute_mlwe_pke_security_level(params: MLWEPKEParams) -> dict:
    """
    Estimate the security level of an MLWE-based PKE parameter set.

    Constructs an equivalent LWE (Learning With Errors) instance based on the
    given parameters and runs the lattice estimator to assess security against
    known attacks such as BKZ, Dual, and Primal attacks.

    The MLWE-based PKE instance is modeled as an LWE problem where:
    - The dimension n equals the total number of ring coefficients (k * d)
    - The modulus q is inherited from the scheme parameters
    - Both the secret and error distributions are centered binomial with parameter eta1
    - The number of equations m equals the dimension n (square LWE system)

    Args:
        params: MLWE-based PKE parameters containing:
            - k: Module dimension (number of polynomials)
            - d: Polynomial degree (coefficients per polynomial)
            - q: The modulus of the ring R_q = Z_q[X]/(X^d + 1)
            - eta1: The parameter for the centered binomial distribution
                    used during key generation

    Returns:
        Dictionary containing security estimates with keys corresponding to
        different lattice attack models (e.g., 'bkz', 'dual', 'primal', etc.)
        and their estimated bit security levels.

    Raises:
        ValueError: If the parameters are invalid.
        RuntimeError: If the estimator cannot complete the security analysis.

    Notes:
        - The security level is estimated against known attacks in the lattice-based
          cryptanalysis literature.
        - Results depend on the lattice estimator version and its internal models.
        - This is a rough/conservative estimate; actual security may vary based on
          optimized attack implementations.
    """
    print("=" * 60)
    print("MLWE-based PKE Security Estimation")
    print("=" * 60)

    # Construct equivalent LWE instance from the scheme parameters.
    # The dimension n = total number of ring coefficients.
    # Each polynomial has degree d, and there are k polynomials in the module.
    lwe_instance = LWEParameters(
        n=params.k * params.d,  # Total dimension: module dimension * polynomial degree
        q=params.q,  # Scheme modulus
        Xs=ND.CenteredBinomial(params.eta1),  # Secret distribution (centered binomial)
        Xe=ND.CenteredBinomial(params.eta1),  # Error distribution (centered binomial)
        m=params.k * params.d,  # Number of equations (square LWE system)
    )

    # Run security estimation against known lattice attacks.
    # This performs a rough analysis of the security level.
    return LWE.estimate.rough(lwe_instance)