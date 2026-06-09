"""
MLWE-based PKE parameter definitions and dataclass.

This module defines the MLWEPKEParams dataclass, which encapsulates the
parameters required for MLWE-based public-key encryption analysis, including
modulus, module dimension, polynomial degree, noise parameters, and ciphertext
compression parameters.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MLWEPKEParams:
    """
    MLWE-based public-key encryption parameters.

    Encapsulates the cryptographic parameters required for an MLWE-based PKE
    parameter set. Different parameter sets may use different module dimensions,
    noise parameters, and compression parameters to target different security_analysis and
    performance tradeoffs.

    Attributes:
        q: The modulus of the ring R_q = Z_q[X]/(X^d + 1).
           Must be greater than 1; in many lattice-based schemes it is prime.

        k: Module dimension, i.e., the number of polynomials in the module.
           This determines the module rank. Common example values are 2, 3,
           or 4, depending on the target security_analysis level.

        eta1: Noise parameter for the centered binomial distribution used during
              key generation when sampling the secret vector s and error vector e
              in the module learning with errors problem.

        eta2: Noise parameter for the centered binomial distribution used during
              encryption when sampling ephemeral secret and error values.

        du: Compression parameter for the ciphertext vector u.
            Specifies the number of bits used to represent each coefficient of u
            after compression.

        dv: Compression parameter for the ciphertext component v.
            Specifies the number of bits used to represent each coefficient of v
            after compression.

        d: Polynomial degree, i.e., the number of coefficients per polynomial.
           Each polynomial in the module has degree d - 1.

    Examples:
        >>> # Example low-rank parameter set
        >>> params_small = MLWEPKEParams(
        ...     q=3329,
        ...     k=2,
        ...     eta1=3,
        ...     eta2=2,
        ...     du=10,
        ...     dv=4,
        ...     d=256,
        ... )

        >>> # Example medium-rank parameter set
        >>> params_medium = MLWEPKEParams(
        ...     q=3329,
        ...     k=3,
        ...     eta1=2,
        ...     eta2=2,
        ...     du=10,
        ...     dv=4,
        ...     d=256,
        ... )

        >>> # Example high-rank parameter set
        >>> params_large = MLWEPKEParams(
        ...     q=3329,
        ...     k=4,
        ...     eta1=2,
        ...     eta2=2,
        ...     du=11,
        ...     dv=5,
        ...     d=256,
        ... )

    Notes:
        - This dataclass is immutable (frozen=True) to prevent accidental
          parameter modification after instantiation.
        - The slots=True optimization reduces memory usage for parameter objects.
        - All parameters are fixed at scheme initialization and remain constant
          throughout key generation, encryption, and decryption analysis.
        - Parameter validation should be performed externally if needed.
    """

    q: int
    k: int
    eta1: int
    eta2: int
    du: int
    dv: int
    d: int