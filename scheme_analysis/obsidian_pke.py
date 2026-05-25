from scheme_analysis.security.mlkem_pke import *
from scheme_analysis.parameter_selection.mlkem_pke import *
from error_term_analyzer.mlkem_pke import *

if __name__ == "__main__":
    param_set = MLKEMParams(
        q=1<<20,
        k=4,
        eta1=2,
        eta2=2,
        du=18,
        dv=12,
        d=256,
    )
    sk_size, pk_size, cipher_size = compute_mlkem_key_ciphertext_size(param_set)

    print(param_set)
    print()
    print(f"|sk|: {sk_size} bytes")
    print(f"|pk|: {pk_size} bytes")
    print(f"|c|: {cipher_size} bytes")

    # print("\n")
    compute_mlkem_security_level(param_set)
    #
    print("\n")
    compute_mlkem_decryption_error(param_set, interactive=True)
