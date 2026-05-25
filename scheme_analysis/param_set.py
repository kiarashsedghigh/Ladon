"""
ML-KEM parameter definitions and dataclass.

This module defines the MLKEMParams dataclass which encapsulates all parameters
required for ML-KEM (Module-Lattice-Based Key-Encapsulation Mechanism) operations,
including modulus, module dimension, polynomial degree, and compression parameters.

References:
    FIPS 203: Module-Lattice-Based Key-Encapsulation Mechanism Standard
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MLKEMParams:
    """
    ML-KEM scheme parameters.

    Encapsulates all cryptographic parameters required for ML-KEM operations.
    Each parameter set (ML-KEM-512, ML-KEM-768, ML-KEM-1024) uses different values
    to achieve different security levels.

    Attributes:
        q: The modulus of the ring R_q = Z_q[X]/(X^d + 1).
           Standard value: 3329. Must be a prime number.

        k: Module dimension - the number of polynomials in the module.
           Determines the module rank; k ∈ {2, 3, 4} for standard ML-KEM variants.
           - ML-KEM-512: k = 2
           - ML-KEM-768: k = 3
           - ML-KEM-1024: k = 4

        eta1: Standard deviation parameter for the centered binomial distribution
              used during key generation for sampling the secret vector s and error_term_analyzer
              vector e in the module learning with errors problem.
              Valid values: 2 or 3 depending on the security level.

        eta2: Standard deviation parameter for the centered binomial distribution
              used during encapsulation for sampling error_term_analyzer vectors.
              Typically smaller than eta1; valid values: 2 or 3.

        du: Compression parameter for the vector u (the encrypted module element).
            Specifies the number of bits used to represent each coefficient
            of u after compression during encapsulation.
            Typical values: 10 or 11 bits per coefficient.

        dv: Compression parameter for the shared secret value v.
            Specifies the number of bits used to represent v after compression.
            Typical values: 4 or 5 bits per coefficient.

        d: Polynomial degree - the number of coefficients per polynomial.
           Standard value: 256. Each polynomial in the module has degree d-1.

    Examples:
        >>> # ML-KEM-512 parameters
        >>> mlkem512 = MLKEMParams(
        ...     q=3329,
        ...     k=2,
        ...     eta1=3,
        ...     eta2=2,
        ...     du=10,
        ...     dv=4,
        ...     d=256
        ... )

        >>> # ML-KEM-768 parameters
        >>> mlkem768 = MLKEMParams(
        ...     q=3329,
        ...     k=3,
        ...     eta1=2,
        ...     eta2=2,
        ...     du=10,
        ...     dv=4,
        ...     d=256
        ... )

        >>> # ML-KEM-1024 parameters
        >>> mlkem1024 = MLKEMParams(
        ...     q=3329,
        ...     k=4,
        ...     eta1=2,
        ...     eta2=2,
        ...     du=11,
        ...     dv=5,
        ...     d=256
        ... )

    Notes:
        - This dataclass is immutable (frozen=True) to prevent accidental parameter
          modification after instantiation.
        - The slots=True optimization reduces memory usage for parameter objects.
        - All parameters are fixed at scheme initialization and remain constant
          throughout all ML-KEM operations (KeyGen, Encaps, Decaps).
        - Parameter validation should be performed externally if needed.
    """

    q: int
    k: int
    eta1: int
    eta2: int
    du: int
    dv: int
    d: int


# Standard ML-KEM parameter sets (optional - for reference)
MLKEM512_PARAMS = MLKEMParams(
    q=3329,
    k=2,
    eta1=3,
    eta2=2,
    du=10,
    dv=4,
    d=256,
)
"""ML-KEM-512: NIST security strength category 1 (≈128-bit equivalent)."""

MLKEM768_PARAMS = MLKEMParams(
    q=3329,
    k=3,
    eta1=2,
    eta2=2,
    du=10,
    dv=4,
    d=256,
)
"""ML-KEM-768: NIST security strength category 3 (≈192-bit equivalent)."""

MLKEM1024_PARAMS = MLKEMParams(
    q=3329,
    k=4,
    eta1=2,
    eta2=2,
    du=11,
    dv=5,
    d=256,
)
"""ML-KEM-1024: NIST security strength category 5 (≈256-bit equivalent)."""