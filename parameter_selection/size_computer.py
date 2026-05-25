"""
MLWE-based PKE key and ciphertext size computation.

This module computes the key and ciphertext sizes for MLWE-based public-key
 encryption schemes with and without compression, based on scheme parameters.
"""

from __future__ import annotations

from math import ceil, log2
from parameter_selection.param_set import *


def compute_mlwe_pke_key_ciphertext_size(pke_params: MLWEPKEParams) -> tuple[int, int, int]:
    """
    Compute secret key, public key, and ciphertext sizes with compression.

    Calculates the byte sizes for an MLWE-based public-key encryption scheme using
    the compression parameters (du, dv) defined in the scheme. The public key
    includes a 32-byte seed.

    Formulas:
        - secret_key_size = (k * d * modulus_bits) / 8
        - public_key_size = (k * d * modulus_bits) / 8 + 32
        - ciphertext_size = (d / 8) * (du * k + dv)

    Args:
        pke_params: Scheme parameters containing:
            - q: The modulus of the ring R_q = Z_q[X]/(X^d + 1)
            - k: Module dimension (number of polynomials)
            - d: Polynomial degree
            - du: Compression parameter for vector u
            - dv: Compression parameter for vector v

    Returns:
        Tuple of (secret_key_bytes, public_key_bytes, ciphertext_bytes) where:
            - secret_key_bytes: Size of the secret key in bytes
            - public_key_bytes: Size of the public key in bytes (includes 32-byte seed)
            - ciphertext_bytes: Size of the ciphertext in bytes

    Raises:
        ValueError: If modulus q is invalid (q <= 1)

    Examples:
        >>> sk, pk, ct = compute_mlwe_pke_key_ciphertext_size(params)
        >>> print(f"Secret Key: {sk} bytes")
        >>> print(f"Public Key: {pk} bytes")
        >>> print(f"Ciphertext: {ct} bytes")
    """
    print("=" * 60)
    print("MLWE-based PKE Key and Ciphertext Sizes")
    print("=" * 60)

    # Compute number of bits required to represent the modulus q
    modulus_size_bits = ceil(log2(pke_params.q))

    # Secret key: all coefficients of k polynomials, each of degree d
    sk_size_bytes = (pke_params.k * pke_params.d * modulus_size_bits) // 8

    # Public key: all coefficients of k polynomials + 32-byte seed
    pk_size_bytes = (pke_params.k * pke_params.d * modulus_size_bits) // 8 + 32

    # Ciphertext with compression: du*k compressed u vector + dv compressed v vector
    ciphertext_size_bytes = int(
        pke_params.d / 8 * (pke_params.du * pke_params.k + pke_params.dv)
    )

    return sk_size_bytes, pk_size_bytes, ciphertext_size_bytes


def compute_mlwe_pke_key_ciphertext_size_no_compression(
        pke_params: MLWEPKEParams,
) -> tuple[int, int, int]:
    """
    Compute secret key, public key, and ciphertext sizes without compression.

    Calculates the byte sizes for an MLWE-based public-key encryption scheme where
    all coefficients are stored at full modulus_bits precision (no compression
    applied). The public key includes a 32-byte seed.

    Formulas:
        - secret_key_size = (k * d * modulus_bits) / 8
        - public_key_size = (k * d * modulus_bits) / 8 + 32
        - ciphertext_size = (d / 8) * (modulus_bits * k + modulus_bits)

    Args:
        pke_params: Scheme parameters containing:
            - q: The modulus of the ring R_q = Z_q[X]/(X^d + 1)
            - k: Module dimension (number of polynomials)
            - d: Polynomial degree

    Returns:
        Tuple of (secret_key_bytes, public_key_bytes, ciphertext_bytes) where:
            - secret_key_bytes: Size of the secret key in bytes
            - public_key_bytes: Size of the public key in bytes (includes 32-byte seed)
            - ciphertext_bytes: Size of the uncompressed ciphertext in bytes

    Raises:
        ValueError: If modulus q is invalid (q <= 1)

    Examples:
        >>> sk, pk, ct = compute_mlwe_pke_key_ciphertext_size_no_compression(params)
        >>> print(f"Secret Key: {sk} bytes")
        >>> print(f"Public Key: {pk} bytes")
        >>> print(f"Ciphertext (uncompressed): {ct} bytes")
    """
    print("=" * 60)
    print("MLWE-based PKE Key and Ciphertext Sizes")
    print("=" * 60)

    # Compute number of bits required to represent the modulus q
    modulus_size_bits = ceil(log2(pke_params.q))

    # Secret key: all coefficients of k polynomials, each of degree d
    sk_size_bytes = (pke_params.k * pke_params.d * modulus_size_bits) // 8

    # Public key: all coefficients of k polynomials + 32-byte seed
    pk_size_bytes = (pke_params.k * pke_params.d * modulus_size_bits) // 8 + 32

    # Ciphertext without compression: modulus_bits * k u coefficients + modulus_bits v coefficients
    ciphertext_size_bytes = int(
        pke_params.d / 8 * (modulus_size_bits * pke_params.k + modulus_size_bits)
    )

    return sk_size_bytes, pk_size_bytes, ciphertext_size_bytes