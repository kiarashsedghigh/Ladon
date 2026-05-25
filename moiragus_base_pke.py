from parameter_selection import *
from error_term_analyzer import *
from security_analysis import *

if __name__ == "__main__":
    params = MLWEPKEParams(
        q=1<<20,
        k=4,
        eta1=2,
        eta2=2,
        du=18,
        dv=12,
        d=256,
    )
    sk_size, pk_size, cipher_size = compute_mlwe_pke_key_ciphertext_size(params)

    print(params)
    print()
    print(f"|sk|: {sk_size} bytes")
    print(f"|pk|: {pk_size} bytes")
    print(f"|c|: {cipher_size} bytes")

    print("\n")
    compute_mlwe_pke_security_level(params)

    print("\n")
    compute_mlwe_pke_decryption_error(params, interactive=True)
