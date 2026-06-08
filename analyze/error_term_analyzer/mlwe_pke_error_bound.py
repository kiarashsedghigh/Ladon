from __future__ import annotations

from analyze.error_term_analyzer.compression_error_bound import *
from analyze.error_term_analyzer.probability_distribution import *
from analyze.parameter_selection.param_set import MLWEPKEParams


def _print_total_error_bound(total_error_coeff, error_bound, d):
    print(f"Results for bound {error_bound}:")

    prob_coeff_success = total_error_coeff.probability_in_range(error_bound)

    prob_coeff_error = Decimal('1.0') - prob_coeff_success

    # Compute log2 using Decimal arithmetic: log2(x) = ln(x) / ln(2)
    if prob_coeff_error > 0:
        coeff_error_bits = -prob_coeff_error.ln() / Decimal(2).ln()
        print(f"  Single coefficient error bits: {float(coeff_error_bits):.4f} bits")
    else:
        print("  Single coefficient error bits: infinity (probability too small)")

    # Apply union bound over all d coefficients
    prob_total_error = Decimal(d) * prob_coeff_error

    if prob_total_error > 0:
        total_error_bits = -prob_total_error.ln() / Decimal(2).ln()
        print(f"  Total decryption error bits: {float(total_error_bits):.4f} bits")
    else:
        print("  Total decryption error bits: infinity (probability too small)")


def compute_mlwe_pke_decryption_error(params: MLWEPKEParams, interactive=False):
    print("=" * 60)
    print("MLWE-based PKE Error Term: rᵀe + e₂ + e′ − (e₁ + e″)ᵀs")
    print("=" * 60)

    print("Sampling base coefficient distributions...")
    r_coeff = sample_binomial_distribution(params.eta1)
    e_coeff = sample_binomial_distribution(params.eta1)
    s_coeff = sample_binomial_distribution(params.eta1)
    e1_coeff = sample_binomial_distribution(params.eta2)
    e2_coeff = sample_binomial_distribution(params.eta2)

    # Compute rᵀe term: sum of k*d products
    print("Computing rᵀe term (sum of k*d products)...")
    r_transpose_e = multiply_probability_distributions(r_coeff, e_coeff) ** (params.k * params.d)

    # Compute e₁ᵀs term: sum of k*d products
    print("Computing e₁ᵀs term (sum of k*d products)...")
    e1_transpose_s = multiply_probability_distributions(e1_coeff, s_coeff) ** (params.k * params.d)

    # Compression error e′ from v compression
    print("Computing e′ compression error (dv-bit)...")
    compression_error_v = compute_modulus_reduction_error_distribution(
        params.q,
        1 << params.dv,
        use_parallel=True,
    )

    # Compression error e″ from u compression
    print("Computing e″ compression error (du-bit)...")
    compression_error_u = compute_modulus_reduction_error_distribution(
        params.q,
        1 << params.du,
        use_parallel=True,
    )

    # Compute e″ᵀs term: sum of k*d products
    print("Computing e″ᵀs term (sum of k*d products)...")
    compression_u_transpose_s = multiply_probability_distributions(compression_error_u, s_coeff) ** (
        params.k * params.d
    )

    # Total error: rᵀe + e₂ + e′ − e₁ᵀs − e″ᵀs
    print("Computing total decryption error distribution...")
    total_error_coeff = (
        r_transpose_e
        * e2_coeff
        * e1_transpose_s
        * compression_error_v
        * compression_u_transpose_s
    )

    print("Computing error/success probabilities...")

    error_bound = 0
    # Success probability for a single coefficient
    if interactive:
        while True:
            new_bound = (input("Enter error bound: "))
            if new_bound == "q":
                return int(error_bound)

            error_bound = int(new_bound)
            _print_total_error_bound(total_error_coeff, error_bound, params.d)

            print()
    else:

        error_bound = params.q // 4 - 1
        _print_total_error_bound(total_error_coeff, error_bound, params.d)


def compute_mlwe_pke_decryption_error_no_compression(params: MLWEPKEParams, interactive=False):
    print("=" * 60)
    print("MLWE-based PKE Error Term without Compression: rᵀe + e₂ − e₁ᵀs")
    print("=" * 60)

    print("Sampling base coefficient distributions...")
    r_coeff = sample_binomial_distribution(params.eta1)
    e_coeff = sample_binomial_distribution(params.eta1)
    s_coeff = sample_binomial_distribution(params.eta1)
    e1_coeff = sample_binomial_distribution(params.eta2)
    e2_coeff = sample_binomial_distribution(params.eta2)

    # Compute rᵀe term: sum of k*d products
    print("Computing rᵀe term (sum of k*d products)...")
    r_transpose_e = multiply_probability_distributions(r_coeff, e_coeff) ** (params.k * params.d)

    # Compute e₁ᵀs term: sum of k*d products
    print("Computing e₁ᵀs term (sum of k*d products)...")
    e1_transpose_s = multiply_probability_distributions(e1_coeff, s_coeff) ** (params.k * params.d)

    # Total error: rᵀe + e₂ − e₁ᵀs
    print("Computing total decryption error distribution...")
    total_error_coeff = r_transpose_e * e2_coeff * e1_transpose_s

    print("Computing error/success probabilities...")

    # Success probability for a single coefficient
    if interactive:
        while True:
            error_bound = int(input("Enter error bound: "))
            _print_total_error_bound(total_error_coeff, error_bound, params.d)

            print()
    else:
        error_bound = params.q // 4 - 1
        _print_total_error_bound(total_error_coeff, error_bound, params.d)