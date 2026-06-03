from parameter_selection import *
from error_term_analyzer import *
from security_analysis import *

from math import ceil, log2
from success_amplifier import run_amplification
from communication_cost import cost_scheme_passive, cost_scheme_active

# def compute_passive_active_scheme_crypto_sizes_passive(pke_params: MLWEPKEParams) -> tuple[int, int, int]:
#     modulus_size_bits = ceil(log2(pke_params.q))
#
#     # Secret key: all coefficients of k polynomials, each of degree d
#     sk_size_bytes = (pke_params.k * pke_params.d * modulus_size_bits) // 8
#
#     # Public key: all coefficients of k polynomials + 32-byte seed
#     pk_size_bytes = (pke_params.k * pke_params.d * modulus_size_bits) // 8 + 32
#
#     # Ciphertext with compression: du*k compressed u vector + dv compressed v vector
#     ciphertext_size_bytes = int(
#         pke_params.d / 8 * (pke_params.du * pke_params.k + pke_params.dv)
#     )
#
#     return sk_size_bytes, pk_size_bytes, ciphertext_size_bytes





def compute_active_scheme_crypto_sizes(pke_params: MLWEPKEParams, statistical_sec) -> tuple[int, int, int, int, int]:
    modulus_size_bits = ceil(log2(pke_params.q))

    # Secret key: all coefficients of k polynomials, each of degree d
    sk_size_bytes = (pke_params.k * pke_params.d * ceil(log2(2 * pke_params.eta1 + 1))) // 8
    sk_size_over_q_bytes = (pke_params.k * pke_params.d * modulus_size_bits) // 8

    # Public key: all coefficients of k polynomials + 32-byte seed
    pk_size_bytes = (pke_params.k * pke_params.d * modulus_size_bits) // 8 + 32

    # Ciphertext with compression: du*k compressed u vector + dv compressed v vector
    ciphertext_size_bytes = int(
        pke_params.d / 8 * (pke_params.du * pke_params.k + pke_params.dv)
    )

    # Sk_share size over larger ring
    sk_share_size_bytes = int (pke_params.k * pke_params.d * (modulus_size_bits + statistical_sec) // 8)

    return sk_size_bytes, sk_size_over_q_bytes, sk_share_size_bytes, pk_size_bytes, ciphertext_size_bytes




def analyze_active_128():
    params = MLWEPKEParams(
        q=1<<27,
        k=5,
        eta1=2,
        eta2=2,
        du=25,
        dv=19,
        d=256,
    )
    statistical_sec = 40
    sk_size, sk_over_q_size, sk_share_size, pk_size, cipher_size = compute_active_scheme_crypto_sizes(params, statistical_sec)

    print(params)
    print()
    print(f"|sk|: {sk_size} bytes")
    print(f"|sk| over q: {sk_over_q_size} bytes")
    print(f"|sk_share|: {sk_share_size} bytes")
    print(f"|mac_share| ... storage: {sk_share_size} bytes")  # // mac and sk shares are in the same domain
    print(f"|pk|: {pk_size} bytes")
    print(f"|c|: {cipher_size} bytes")

    print("\n")
    compute_mlwe_pke_security_level(params)

    print("\n")
    error_norm = compute_mlwe_pke_decryption_error(params, interactive=True)

    error_bits = -(ceil(log2(params.q)) - 1 - (ceil(log2(params.d)) + ceil(log2(error_norm))))
    print(f"Threshold Failure (N |e| / mu) as bits: {error_bits} \n")
    print("\n\n")

    ## Print success amplifier
    run_amplification(error_bits)


    ## Ask for parameter l (number of parallel decryptions) to achieve a target failure probability
    param_l = int(input("Enter l: "))

    ## Run communication analysis
    cost_scheme_active(
        log2_q   = ceil(log2(params.q)),            # placeholder: q = 2^32
        ell      = param_l,             # placeholder
        d        = params.d,
        ct_bytes = cipher_size,          # placeholder
        lambda_s = statistical_sec,
    )



def analyze_active_256():
    params = MLWEPKEParams(
        q=1<<27,
        k=9,
        eta1=2,
        eta2=2,
        du=25,
        dv=18,
        d=256,
    )
    statistical_sec = 40
    sk_size, sk_over_q_size, sk_share_size, pk_size, cipher_size = compute_active_scheme_crypto_sizes(params, statistical_sec)

    print(params)
    print()
    print(f"|sk|: {sk_size} bytes")
    print(f"|sk| over q: {sk_over_q_size} bytes")
    print(f"|sk_share|: {sk_share_size} bytes")
    print(f"|mac_share| ... storage: {sk_share_size} bytes")  # // mac and sk shares are in the same domain
    print(f"|pk|: {pk_size} bytes")
    print(f"|c|: {cipher_size} bytes")

    print("\n")
    compute_mlwe_pke_security_level(params)

    print("\n")
    error_norm = compute_mlwe_pke_decryption_error(params, interactive=True)

    error_bits = -(ceil(log2(params.q)) - 1 - (ceil(log2(params.d)) + ceil(log2(error_norm))))
    print(f"Threshold Failure (N |e| / mu) as bits: {error_bits} \n")
    print("\n\n")

    ## Print success amplifier
    run_amplification(error_bits)


    ## Ask for parameter l (number of parallel decryptions) to achieve a target failure probability
    param_l = int(input("Enter l: "))

    ## Run communication analysis
    cost_scheme_active(
        log2_q   = ceil(log2(params.q)),            # placeholder: q = 2^32
        ell      = param_l,             # placeholder
        d        = params.d,
        ct_bytes = cipher_size,          # placeholder
        lambda_s = statistical_sec,
    )

if __name__ == "__main__":
    # analyze_active_128()
    #
    #
    analyze_active_256()