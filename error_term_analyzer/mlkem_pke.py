from error_term_analyzer.compression_error_bound import *
from error_term_analyzer.prob_dist import *
from scheme_analysis.param_set import MLKEMParams
from decimal import Decimal


def nice_print_total_error_bound(total_error_coeff, error_bound, d):
    print(f"Results for bound {error_bound}:")

    prob_coeff_success = total_error_coeff.probability_in_range(error_bound)

    prob_coeff_error = Decimal('1.0') - prob_coeff_success

    # Compute log2 using Decimal arithmetic: log2(x) = ln(x) / ln(2)
    if prob_coeff_error > 0:
        coeff_error_bits = -prob_coeff_error.ln() / Decimal(2).ln()
        print(f"  Single coefficient error_term_analyzer bits: {float(coeff_error_bits):.4f} bits")
    else:
        print(f"  Single coefficient error_term_analyzer bits: infinity (probability too small)")

    # Apply union bound over all d coefficients
    prob_total_error = Decimal(d) * prob_coeff_error

    if prob_total_error > 0:
        total_error_bits = -prob_total_error.ln() / Decimal(2).ln()
        print(f"  Total decryption error_term_analyzer bits: {float(total_error_bits):.4f} bits")
    else:
        print(f"  Total decryption error_term_analyzer bits: infinity (probability too small)")


def compute_mlkem_decryption_error(mlkem_params: MLKEMParams, interactive=False):
    print("=" * 60)
    print("ML-KEM Error Term: rᵀe + e₂ + e′ − (e₁ + e″)ᵀs")
    print("=" * 60)

    print("Sampling base coefficient distributions...")
    r_coeff = sample_binomial_distribution(mlkem_params.eta1)
    e_coeff = sample_binomial_distribution(mlkem_params.eta1)
    s_coeff = sample_binomial_distribution(mlkem_params.eta1)
    e1_coeff = sample_binomial_distribution(mlkem_params.eta2)
    e2_coeff = sample_binomial_distribution(mlkem_params.eta2)

    # Compute rᵀe term: sum of k*d products
    print("Computing rᵀe term (sum of k*d products)...")
    r_transpose_e = multiply_probability_distributions(r_coeff, e_coeff) ** (mlkem_params.k * mlkem_params.d)

    # Compute e₁ᵀs term: sum of k*d products
    print("Computing e₁ᵀs term (sum of k*d products)...")
    e1_transpose_s = multiply_probability_distributions(e1_coeff, s_coeff) ** (mlkem_params.k * mlkem_params.d)

    # Compression error_term_analyzer e′ from v compression
    print("Computing e′ compression error_term_analyzer (dv-bit)...")
    compression_error_v = compute_compression_error_distribution(mlkem_params.q,
                                                                 1 << mlkem_params.dv,
                                                                 use_parallel=True)

    # Compression error_term_analyzer e″ from u compression
    print("Computing e″ compression error_term_analyzer (du-bit)...")
    compression_error_u = compute_compression_error_distribution(mlkem_params.q,
                                                                 1 << mlkem_params.du,
                                                                 use_parallel=True)

    # Compute e″ᵀs term: sum of k*d products
    print("Computing e″ᵀs term (sum of k*d products)...")
    compression_u_transpose_s = multiply_probability_distributions(compression_error_u, s_coeff) ** (
            mlkem_params.k * mlkem_params.d)

    # Total error_term_analyzer: rᵀe + e₂ + e′ − e₁ᵀs − e″ᵀs
    print("Computing total decryption error_term_analyzer distribution...")
    total_error_coeff = (r_transpose_e *
                         e2_coeff *
                         e1_transpose_s *
                         compression_error_v *
                         compression_u_transpose_s)

    print("Computing error_term_analyzer/success probabilities...")

    # Success probability for a single coefficient
    if interactive:
        while True:
            error_bound = int(input("Enter error_term_analyzer bound: "))
            nice_print_total_error_bound(total_error_coeff, error_bound, mlkem_params.d)

            print()
    else:
        error_bound = mlkem_params.q // 4 - 1
        nice_print_total_error_bound(total_error_coeff, error_bound, mlkem_params.d)


def compute_mlkem_decryption_error_no_compression(mlkem_params: MLKEMParams, interactive=False):
    print("=" * 60)
    print("ML-KEM Error Term: rᵀe + e₂ + e′ − (e₁ + e″)ᵀs")
    print("=" * 60)

    print("Sampling base coefficient distributions...")
    r_coeff = sample_binomial_distribution(mlkem_params.eta1)
    e_coeff = sample_binomial_distribution(mlkem_params.eta1)
    s_coeff = sample_binomial_distribution(mlkem_params.eta1)
    e1_coeff = sample_binomial_distribution(mlkem_params.eta2)
    e2_coeff = sample_binomial_distribution(mlkem_params.eta2)

    # Compute rᵀe term: sum of k*d products
    print("Computing rᵀe term (sum of k*d products)...")
    r_transpose_e = multiply_probability_distributions(r_coeff, e_coeff) ** (mlkem_params.k * mlkem_params.d)

    # Compute e₁ᵀs term: sum of k*d products
    print("Computing e₁ᵀs term (sum of k*d products)...")
    e1_transpose_s = multiply_probability_distributions(e1_coeff, s_coeff) ** (mlkem_params.k * mlkem_params.d)

    # Total error_term_analyzer: rᵀe + e₂ − e₁ᵀs
    print("Computing total decryption error_term_analyzer distribution...")
    total_error_coeff = r_transpose_e * e2_coeff * e1_transpose_s

    print("Computing error_term_analyzer/success probabilities...")

    # Success probability for a single coefficient
    if interactive:
        while True:
            error_bound = int(input("Enter error_term_analyzer bound: "))
            nice_print_total_error_bound(total_error_coeff, error_bound, mlkem_params.d)

            print()
    else:
        error_bound = mlkem_params.q // 4 - 1
        nice_print_total_error_bound(total_error_coeff, error_bound, mlkem_params.d)
