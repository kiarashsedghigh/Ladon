//! Negacyclic matrix construction over Z[X]/(X^256 + 1), for the Z_{2^k} branch.
//!
//! In the prime build, polynomial multiplication uses the NTT. Over a power-of-
//! two modulus 2^k there is no NTT, so multiplication by a fixed polynomial a
//! is realized as a matrix-vector product:
//!
//!     (a * b)  ==  Negacyclic(a) @ b        (mod X^256 + 1)
//!
//! where Negacyclic(a) is the 256x256 matrix
//!
//!     M[k][j] =  a[k-j]            if k >= j
//!             = -a[k-j+256]        if k <  j      (sign flip: X^256 = -1)
//!
//! Element type: i128. This was BigInt at first, but ndarray's `.dot()`
//! requires `LinalgScalar`, which requires `Copy` — BigInt is heap-backed and
//! not Copy, so Array2<BigInt>::dot doesn't compile. i128 is Copy, satisfies
//! LinalgScalar, and gives plenty of headroom:
//!
//!     k=20 base : products ~40b, row-sum over 512 entries ~49b  (safe)
//!     k up to ~56b: still safe (product ~112b, +log2(256K) ~9b => 121b in i128)
//!
//! Final reduction is `rem_euclid` against 2^k (in i128), then a `to_u32`
//! narrowing for storage back into `Ring.data`.

use ndarray::{Array1, Array2};

use crate::ring::{Matrix, Ring, Vector};

pub const N: usize = 256;

/// Build the 256x256 negacyclic matrix for polynomial `a` (given by its 256
/// coefficients).
pub fn negacyclic_matrix(coeffs: &[i128; N]) -> Array2<i128> {
    let mut m = Array2::<i128>::zeros((N, N));
    for k in 0..N {
        for j in 0..N {
            if k >= j {
                m[[k, j]] = coeffs[k - j];
            } else {
                // wrapped term, negated (X^256 = -1).
                // index = k - j + N, but k < j so k - j underflows usize.
                // Reorder as N - (j - k); since j > k, (j - k) >= 1, so the
                // index is in [1, N-1], non-negative.
                m[[k, j]] = -coeffs[N - (j - k)];
            }
        }
    }
    m
}

/// Convert a Ring's coefficients (u32, coefficient form) into an i128 array.
/// NOTE: pass a Ring in coefficient (Degree255) form. NTT-form values are not
/// the polynomial's coefficients and would build a meaningless matrix.
pub fn ring_to_i128_coeffs(r: &Ring) -> [i128; N] {
    core::array::from_fn(|i| r.data[i] as i128)
}

/// Negacyclic matrix straight from a Ring (coefficient form).
pub fn ring_to_negacyclic(r: &Ring) -> Array2<i128> {
    let coeffs = ring_to_i128_coeffs(r);
    negacyclic_matrix(&coeffs)
}

/// Stack a K-length vector of polynomials into one column vector of length
/// 256*K (i128), preserving coefficient order: block i occupies rows
/// [256*i, 256*(i+1)).
pub fn vector_to_column<const K: usize>(v: &Vector<K>) -> Array1<i128> {
    let mut col = Array1::<i128>::zeros(N * K);
    for i in 0..K {
        for c in 0..N {
            col[N * i + c] = v.data[i].data[c] as i128;
        }
    }
    col
}

/// Inverse of `vector_to_column`: unpack a length-256*K i128 column (assumed
/// already reduced into [0, 2^k)) back into a Vector<K> of coefficient-form
/// Rings. Each entry must fit u32 (k <= 32).
pub fn column_to_vector<const K: usize>(col: &Array1<i128>) -> Vector<K> {
    let mut v = Vector::<K>::new_degree255();
    for i in 0..K {
        for c in 0..N {
            let x = col[N * i + c];
            debug_assert!(x >= 0 && x <= u32::MAX as i128, "coeff out of u32 range: {x}");
            v.data[i].data[c] = x as u32;
        }
    }
    v
}

/// Build the full (256*K) x (256*K) block matrix from a KxK grid of polynomial
/// coefficient sets. `blocks[i][j]` are the 256 coeffs of polynomial A[i][j].
pub fn block_negacyclic_matrix<const K: usize>(
    blocks: &[[[i128; N]; K]; K],
) -> Array2<i128> {
    let dim = N * K;
    let mut big = Array2::<i128>::zeros((dim, dim));
    for bi in 0..K {
        for bj in 0..K {
            let sub = negacyclic_matrix(&blocks[bi][bj]);
            for k in 0..N {
                for j in 0..N {
                    big[[N * bi + k, N * bj + j]] = sub[[k, j]];
                }
            }
        }
    }
    big
}

/// Build the full (256*K) x (256*K) negacyclic block matrix directly from a
/// `Matrix<K>` of Rings (coefficient form). One call -> one big A_hat ready
/// for a single .dot(s_col).
pub fn matrix_to_block_negacyclic<const K: usize>(a: &Matrix<K>) -> Array2<i128> {
    let dim = N * K;
    let mut big = Array2::<i128>::zeros((dim, dim));
    for bi in 0..K {
        for bj in 0..K {
            let sub = ring_to_negacyclic(&a.data[bi][bj]);
            for k in 0..N {
                for j in 0..N {
                    big[[N * bi + k, N * bj + j]] = sub[[k, j]];
                }
            }
        }
    }
    big
}

/// Build the negacyclic block matrix for A^T (matrix-transpose at the BLOCK
/// level): block (i,j) of the result is Negacyclic(A[j][i]), i.e. each block
/// is the un-transposed negacyclic of the swapped index.
///
/// IMPORTANT: do NOT use `.t()` on the result of `matrix_to_block_negacyclic` —
/// `.t()` would also transpose each 256x256 sub-block, but Negacyclic(a) is
/// not symmetric, so that gives wrong answers. The correct transpose at the
/// polynomial level only swaps which polynomial sits in which block slot.
pub fn matrix_to_block_negacyclic_transposed<const K: usize>(a: &Matrix<K>) -> Array2<i128> {
    let dim = N * K;
    let mut big = Array2::<i128>::zeros((dim, dim));
    for bi in 0..K {
        for bj in 0..K {
            // block (bi, bj) gets Negacyclic(A[bj][bi])  <-- indices swapped
            let sub = ring_to_negacyclic(&a.data[bj][bi]);
            for k in 0..N {
                for j in 0..N {
                    big[[N * bi + k, N * bj + j]] = sub[[k, j]];
                }
            }
        }
    }
    big
}

/// Build a 256 x (256*K) row of negacyclic blocks from a length-K vector of
/// polynomials. This is the matrix form of the inner product <v, r>: for each
/// i, place Negacyclic(v_i) as block i along the row. Then
///   row_matrix.dot(&r_col) == sum_i (v_i * r_i)
/// as a single length-256 polynomial — what t^T r evaluates to in encrypt.
pub fn vector_to_row_negacyclic<const K: usize>(v: &Vector<K>) -> Array2<i128> {
    let mut row = Array2::<i128>::zeros((N, N * K));
    for bj in 0..K {
        let sub = ring_to_negacyclic(&v.data[bj]);
        for k in 0..N {
            for j in 0..N {
                row[[k, N * bj + j]] = sub[[k, j]];
            }
        }
    }
    row
}

/// Reduce every entry of an i128 vector into [0, 2^k) by rem_euclid.
/// Handles negatives from the sign flips / CBD noise.
pub fn reduce_mod_2k(v: &mut Array1<i128>, k: u32) {
    let modulus: i128 = 1i128 << k;
    for x in v.iter_mut() {
        *x = x.rem_euclid(modulus);
    }
}

// ---------------------------------------------------------------------------
// Compress / decompress for q = 2^k
//
// Over q = 2^k these collapse to (rounded) bit shifts:
//   compress_D(x)   = round(x * 2^D / 2^k)  mod 2^D
//                   = round(x >> (k - D))
//                   = ((x + 2^(k-D-1)) >> (k-D))  &  ((1<<D) - 1)
//   decompress_D(y) = round(y * 2^k / 2^D)  =  y << (k - D)      (exact)
//
// (Verified against the divide-by-q formula.)
// ---------------------------------------------------------------------------

/// Compress one coefficient from [0, 2^k) to [0, 2^D). D must be <= k.
#[inline]
pub fn compress_2k(x: u32, d: u32, k: u32) -> u32 {
    debug_assert!(d <= k);
    if d == k {
        x & ((1u32 << d) - 1)
    } else {
        let shift = k - d;
        let half = 1u32 << (shift - 1);
        ((x.wrapping_add(half)) >> shift) & ((1u32 << d) - 1)
    }
}

/// Decompress one coefficient from [0, 2^D) back to [0, 2^k). Exact shift.
#[inline]
pub fn decompress_2k(y: u32, d: u32, k: u32) -> u32 {
    debug_assert!(d <= k);
    y << (k - d)
}

/// Compress a Ring (mod 2^k, coefficient form) to D bits/coefficient in place
/// of a new Ring whose coefficients live in [0, 2^D). Returns the compressed
/// Ring (still stored as u32 in the Ring's data field; only the low D bits
/// carry information).
pub fn compress_ring_2k(r: &Ring, d: u32, k: u32) -> Ring {
    let mut out = Ring::ZEROES_DEGREE255;
    for c in 0..N {
        out.data[c] = compress_2k(r.data[c], d, k);
    }
    out
}

/// Decompress a Ring whose coefficients are in [0, 2^D) back to mod-2^k values.
pub fn decompress_ring_2k(r: &Ring, d: u32, k: u32) -> Ring {
    let mut out = Ring::ZEROES_DEGREE255;
    for c in 0..N {
        out.data[c] = decompress_2k(r.data[c], d, k);
    }
    out
}

/// Compress every ring of a Vector<K>.
pub fn compress_vector_2k<const K: usize>(v: &Vector<K>, d: u32, k: u32) -> Vector<K> {
    let mut out = Vector::<K>::new_degree255();
    for i in 0..K {
        out.data[i] = compress_ring_2k(&v.data[i], d, k);
    }
    out
}

/// Decompress every ring of a Vector<K>.
pub fn decompress_vector_2k<const K: usize>(v: &Vector<K>, d: u32, k: u32) -> Vector<K> {
    let mut out = Vector::<K>::new_degree255();
    for i in 0..K {
        out.data[i] = decompress_ring_2k(&v.data[i], d, k);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_negacyclic_wrap_sign_flip() {
        // a = X^1 (coeff[1]=1), b = X^255 (coeff[255]=1)  =>  a*b = X^256 = -1
        let mut a = [0i128; N];
        a[1] = 1;
        let m = negacyclic_matrix(&a);

        let mut b = Array1::<i128>::zeros(N);
        b[255] = 1;
        let c = m.dot(&b);

        assert_eq!(c[0], -1, "X*X^255 should give -1 (the X^256 = -1 wrap)");
        for i in 1..N {
            assert_eq!(c[i], 0);
        }
    }

    #[test]
    fn test_negacyclic_no_wrap() {
        // a = X^1, b = X^3  =>  a*b = X^4 (no wrap), coeff[4] = 1
        let mut a = [0i128; N];
        a[1] = 1;
        let m = negacyclic_matrix(&a);

        let mut b = Array1::<i128>::zeros(N);
        b[3] = 1;
        let c = m.dot(&b);

        assert_eq!(c[4], 1);
        for i in 0..N {
            if i != 4 {
                assert_eq!(c[i], 0);
            }
        }
    }

    #[test]
    fn test_reduce_mod_2k() {
        let k = 20u32;
        let m = 1i128 << k;
        let mut v = Array1::from(vec![-1i128, m, m + 5, 3, -((m / 2) as i128)]);
        reduce_mod_2k(&mut v, k);
        assert_eq!(v[0], m - 1);
        assert_eq!(v[1], 0);
        assert_eq!(v[2], 5);
        assert_eq!(v[3], 3);
        assert_eq!(v[4], m - (m / 2));
    }
}