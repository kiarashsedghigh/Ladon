use std::process::exit;
use crate::crypt;
use crate::params::*;
use crate::ring::*;
use crate::sample;

pub type KpkeEncryptionKey <const K: usize> = (Vector<{K}>, [u8; 32]);
pub type KpkeDecryptionKey <const K: usize> = Vector<{K}>;

pub type KpkeKeyGenOutput <const K: usize> = (KpkeEncryptionKey<{K}>, KpkeDecryptionKey<{K}>);

pub fn key_gen<PARAMS: MlKemParams>() -> KpkeKeyGenOutput<{PARAMS::K}> where
    [(); 960 * PARAMS::K + 32]: ,
    [(); 1920 * PARAMS::K + 96]: ,
    [(); PARAMS::K]: ,
    [(); PARAMS::ETA_2]: ,
    [(); 64 * PARAMS::ETA_1]: ,
    [(); 32 * (PARAMS::D_U * PARAMS::K + PARAMS::D_V)]: ,
{
    // This is the main source of randomness for Party A (he also would've generated a value to use a random implict rejection answer).
    let d = crypt::random_bytes::<32>();

    let (rho, sigma) = crypt::g::<32>(&d);

    let mut n = 0;

    // Our public key, (the bad bases)
    let mut a: Matrix<{PARAMS::K}> = Matrix::new_ntt();

    for i in 0..PARAMS::K {
        for j in 0..PARAMS::K {
            a.data[i][j] = sample::sample_ntt(crypt::XOF::new(&rho, i as u8, j as u8)) // XOF stream is instantied here for each index of the matrix
        }
    }

    // Our secret key
    let mut s = Vector::new_degree255();
    for i in 0..PARAMS::K {
        s.data[i] = sample::sample_poly_cbd::<{PARAMS::ETA_1}>(
            crypt::prf::<{PARAMS::ETA_1}>(&sigma, n)
        );
        n += 1;
    }

    // Our error vector
    let mut e = Vector::new_degree255();
    for i in 0..PARAMS::K {
        e.data[i] = sample::sample_poly_cbd::<{PARAMS::ETA_1}>(
            crypt::prf::<{PARAMS::ETA_1}>(&sigma, n)
        );
        n += 1;
    }

    // NTT both
    let s = s.ntt();
    let e = e.ntt();

    // The encapsulation key we send includes this vector t, our secret linear transformation, with some rror
    let mut t = a.right_vector_multiply(&s);
    t.add(&e);

    // We also send the seed used for A, so the other party can recreate it
    // Our deryption key is just the secret vector
    ((t, rho), s)
}



// ============================================================================
// key_gen for the Z_{2^k} branch (Option A: 2^k math inside, Vector<K> I/O).
//
// Drop-in replacement for the prime kpke::key_gen. Differences vs the prime
// version:
//   * NO NTT anywhere. s, e, A, t all stay in coefficient form.
//   * A is sampled with the same rho + XOF stream, but each coefficient is the
//     low K_BITS bits of a 24-bit XOF draw (uniform mod 2^k, no rejection),
//     instead of prime rejection sampling.
//   * t = A*s + e is computed via NEGACYCLIC MATRIX multiplication (ndarray,
//     i128 element type — see note in negacyclic.rs on why not BigInt), then
//     reduced mod 2^k and packed back into a Vector<K>.
//
// Requires in scope:
//   use ndarray::Array1;
//   use crate::negacyclic;
//   use crate::ring::{Ring, RingRepresentation, Vector};
//   use crate::{crypt, sample};
//   use crate::params::*;
//
// K_BITS: the power-of-two modulus is 2^K_BITS. For now defined locally; move to
// params.rs next to Q when you wire the whole 2^k build.
// ============================================================================

use crate::negacyclic;

/// Sample one ring of A uniformly mod 2^k, in coefficient form, from an XOF
/// stream keyed by (rho, i, j). Each coefficient = low K_BITS bits of a 24-bit
/// (3-byte) XOF draw. Valid for K_BITS <= 24.
// fn sample_uniform_2k(mut xof: crypt::XOF) -> Ring {
//     assert!(K_BITS <= 24, "1-draw sampler needs K_BITS <= 24");
//     let mask: u32 = ((1u64 << K_BITS) - 1) as u32;
//
//     let mut ring = Ring::ZEROES_DEGREE255; // coefficient form, not NTT
//     let mut three = [0u8; 3];
//     for c in 0..256 {
//         xof.get_3_bytes(&mut three);
//         let d = (three[0] as u32) | ((three[1] as u32) << 8) | ((three[2] as u32) << 16);
//         ring.data[c] = d & mask;
//     }
//     ring
// }

/// Sample one ring of A uniformly mod 2^k, in coefficient form, from an XOF
/// stream keyed by (rho, i, j). Each coefficient = low K_BITS bits of a u64
/// drawn from the XOF (two 3-byte reads = 48 bits). Valid for K_BITS <= 48,
/// which covers all sensible PKE settings (k up to ~48 stays well inside
/// the i128 negacyclic-matmul headroom too).
fn sample_uniform_2k(mut xof: crypt::XOF) -> Ring {
    assert!(K_BITS <= 48, "sampler needs K_BITS <= 48 (uses two 3-byte XOF reads)");
    let mask: u64 = (1u64 << K_BITS) - 1;

    let mut ring = Ring::ZEROES_DEGREE255; // coefficient form, not NTT
    let mut three = [0u8; 3];
    for c in 0..256 {
        // First 3 bytes -> low 24 bits.
        xof.get_3_bytes(&mut three);
        let lo = (three[0] as u64) | ((three[1] as u64) << 8) | ((three[2] as u64) << 16);

        // Second 3 bytes -> next 24 bits. Always draw both, regardless of
        // K_BITS, so the XOF stream advances deterministically and the
        // sampler is constant-work per coefficient.
        xof.get_3_bytes(&mut three);
        let hi = (three[0] as u64) | ((three[1] as u64) << 8) | ((three[2] as u64) << 16);

        let d: u64 = lo | (hi << 24);
        ring.data[c] = (d & mask) as u32;
    }
    ring
}

/// Reduce a Ring's coefficients into [0, 2^k) in place (CBD output is reduced
/// mod the prime by sample_poly_cbd; re-canonicalize into 2^k here).
/// CBD values are small and centered; mapping them mod 2^k keeps -y as 2^k - y.
fn reduce_ring_mod_2k(r: &mut Ring) {
    // The sampler returned values mod params::Q (a ~23-bit prime). A negative
    // CBD value -y was stored as Q - y. To re-express mod 2^k we must recover
    // the signed value first, then reduce mod 2^k.
    let q = crate::params::Q;
    let half = q / 2;
    let m: i64 = 1i64 << K_BITS;
    for c in 0..256 {
        let v = r.data[c];
        // signed representative in (-q/2, q/2]
        let signed: i64 = if v > half { v as i64 - q as i64 } else { v as i64 };
        let red = ((signed % m) + m) % m; // rem_euclid mod 2^k
        r.data[c] = red as u32;
    }
}

pub fn key_gen_2k<PARAMS: MlKemParams>() -> KpkeKeyGenOutput<{ PARAMS::K }>
where
    [(); 960 * PARAMS::K + 32]:,
    [(); 1920 * PARAMS::K + 96]:,
    [(); PARAMS::K]:,
    [(); PARAMS::ETA_2]:,
    [(); 64 * PARAMS::ETA_1]:,
    [(); 32 * (PARAMS::D_U * PARAMS::K + PARAMS::D_V)]:,
{
    use ndarray::Array1;

    // Randomness, same structure as the prime version.
    let d = crypt::random_bytes::<32>();
    let (rho, sigma) = crypt::g::<32>(&d);
    let mut n = 0;

    // ---- Public matrix A: sample uniformly mod 2^k, COEFFICIENT form -------
    // Same rho + XOF addressing as before; only the per-coefficient decode
    // changed (mask to K_BITS bits instead of prime rejection).
    let mut a: Matrix<{ PARAMS::K }> = Matrix::new_ntt(); // container; we overwrite + retag
    for i in 0..PARAMS::K {
        for j in 0..PARAMS::K {
            a.data[i][j] = sample_uniform_2k(crypt::XOF::new(&rho, i as u8, j as u8));
        }
    }


    // ---- Secret s and error e: CBD, COEFFICIENT form, reduced mod 2^k ------
    let mut s = Vector::<{ PARAMS::K }>::new_degree255();
    for i in 0..PARAMS::K {
        let mut ring = sample::sample_poly_cbd::<{ PARAMS::ETA_1 }>(
            crypt::prf::<{ PARAMS::ETA_1 }>(&sigma, n),
        );
        reduce_ring_mod_2k(&mut ring);
        s.data[i] = ring;
        n += 1;
    }

    let mut e = Vector::<{ PARAMS::K }>::new_degree255();
    for i in 0..PARAMS::K {
        let mut ring = sample::sample_poly_cbd::<{ PARAMS::ETA_1 }>(
            crypt::prf::<{ PARAMS::ETA_1 }>(&sigma, n),
        );
        reduce_ring_mod_2k(&mut ring);
        e.data[i] = ring;
        n += 1;
    }

    // ---- t = A*s + e  via ONE big negacyclic matrix multiplication --------
    // Build the full (256*K) x (256*K) negacyclic block matrix A_hat, stack s
    // and e into length-256*K columns, do a single matmul, add e, reduce mod
    // 2^k, and unpack back into a Vector<K>.
    let a_hat = negacyclic::matrix_to_block_negacyclic::<{ PARAMS::K }>(&a);
    let s_col: Array1<i128> = negacyclic::vector_to_column::<{ PARAMS::K }>(&s);
    let e_col: Array1<i128> = negacyclic::vector_to_column::<{ PARAMS::K }>(&e);

    let mut t_col = a_hat.dot(&s_col) + e_col; // (A_hat @ s) + e, unreduced
    negacyclic::reduce_mod_2k(&mut t_col, K_BITS); // -> [0, 2^k)

    let t = negacyclic::column_to_vector::<{ PARAMS::K }>(&t_col);

    // Return ((t, rho), s) — same shape as the prime version, all coeff form.
    ((t, rho), s)
}



































pub type Cyphertext<const K: usize, const D_U: usize, const D_V: usize> = (Compressed<{D_U}, Vector<{K}>>, Compressed<{D_V}, Ring>);

pub fn encrypt<PARAMS: MlKemParams>(ek_pke: KpkeEncryptionKey<{PARAMS::K}>, m: Compressed<1,Ring>, rand: [u8; 32]) -> Cyphertext<{PARAMS::K}, {PARAMS::D_U}, {PARAMS::D_V}> where
    [(); PARAMS::K]: ,
    [(); 64 * PARAMS::ETA_1]: ,
    [(); 64 * PARAMS::ETA_2]: ,
    [(); 960 * PARAMS::K + 32]: ,
    [(); PARAMS::D_U]: ,
{
    let mut n = 0;

    let (t, rho) = ek_pke; // rho is the seed for A, the matrix, t comes from KeyGen's computation with their secret

    // Recreate the matrix A
    let mut a: Matrix<{PARAMS::K}> = Matrix::new_ntt();
    for i in 0..PARAMS::K {
        for j in 0..PARAMS::K {
            a.data[i][j] = sample::sample_ntt(crypt::XOF::new(&rho, i as u8, j as u8));
        }
    }

    // Encrpytor's Secret (Equivalent of S in key_gen)
    let mut r: Vector<{PARAMS::K}> = Vector::new_degree255();
    for i in 0..PARAMS::K {
        r.data[i] = sample::sample_poly_cbd::<{PARAMS::ETA_1}>(
            crypt::prf::<{PARAMS::ETA_1}>(&rand, n as u8)
        );
        n += 1;
    }

    // Error vector to be added to R^T * A
    let mut e_1: Vector<{PARAMS::K}> = Vector::new_degree255();
    for i in 0..PARAMS::K {
        e_1.data[i] = sample::sample_poly_cbd::<{PARAMS::ETA_2}>(
            crypt::prf::<{PARAMS::ETA_2}>(&rand, n as u8)
        );
        n += 1;
    }

    // Error vector to be added to the shared key V (R^T * t)
    let e_2 = sample::sample_poly_cbd::<{PARAMS::ETA_2}>(
        crypt::prf::<{PARAMS::ETA_2}>(&rand, n as u8)
    );

    let r = r.ntt();

    // u is the encryptors computation with A and their secret, but this one is left-multiplied
    let mut u = a.left_vector_multiply(&r).inverse_ntt();
    u.add(&e_1);
    let u_compressed = Compressed::<{PARAMS::D_U}, Vector<{PARAMS::K}>>::compress(u);

    let m = m.decompress();

    // v is our shared secret, notice for both parties its approximately rAs.
    let mut v_ntt = r.inner_product(t);
    v_ntt.inverse_ntt().add(&e_2).add(&m);

    let v_compressed = Compressed::<{PARAMS::D_V}, Ring>::compress(v_ntt);

    (u_compressed, v_compressed)
}



pub fn encrypt_2k<PARAMS: MlKemParams>(
    ek_pke: KpkeEncryptionKey<{ PARAMS::K }>,
    m: Compressed<1, Ring>,
    rand: [u8; 32],
) -> Cyphertext<{ PARAMS::K }, { PARAMS::D_U }, { PARAMS::D_V }>
where
    [(); PARAMS::K]:,
    [(); 64 * PARAMS::ETA_1]:,
    [(); 64 * PARAMS::ETA_2]:,
    [(); 960 * PARAMS::K + 32]:,
    [(); PARAMS::D_U]:,
{
    use ndarray::Array1;

    let mut n = 0u8;
    let (t, rho) = ek_pke;

    // ---- Regenerate A mod 2^k (coefficient form) ---------------------------
    let mut a: Matrix<{ PARAMS::K }> = Matrix::new_ntt(); // container; we overwrite
    for i in 0..PARAMS::K {
        for j in 0..PARAMS::K {
            a.data[i][j] = sample_uniform_2k(crypt::XOF::new(&rho, i as u8, j as u8));
        }
    }

    // ---- r, e_1, e_2: CBD then re-reduce mod 2^k ---------------------------
    let mut r = Vector::<{ PARAMS::K }>::new_degree255();
    for i in 0..PARAMS::K {
        let mut ring = sample::sample_poly_cbd::<{ PARAMS::ETA_1 }>(
            crypt::prf::<{ PARAMS::ETA_1 }>(&rand, n),
        );
        reduce_ring_mod_2k(&mut ring);
        r.data[i] = ring;
        n += 1;
    }

    let mut e_1 = Vector::<{ PARAMS::K }>::new_degree255();
    for i in 0..PARAMS::K {
        let mut ring = sample::sample_poly_cbd::<{ PARAMS::ETA_2 }>(
            crypt::prf::<{ PARAMS::ETA_2 }>(&rand, n),
        );
        reduce_ring_mod_2k(&mut ring);
        e_1.data[i] = ring;
        n += 1;
    }

    let mut e_2 = sample::sample_poly_cbd::<{ PARAMS::ETA_2 }>(
        crypt::prf::<{ PARAMS::ETA_2 }>(&rand, n),
    );
    reduce_ring_mod_2k(&mut e_2);

    // ---- u = A^T r + e_1  (one big matmul, transposed at the BLOCK level) --
    let a_hat_t = negacyclic::matrix_to_block_negacyclic_transposed::<{ PARAMS::K }>(&a);
    let r_col: Array1<i128> = negacyclic::vector_to_column::<{ PARAMS::K }>(&r);
    let e1_col: Array1<i128> = negacyclic::vector_to_column::<{ PARAMS::K }>(&e_1);

    let mut u_col = a_hat_t.dot(&r_col) + e1_col;
    negacyclic::reduce_mod_2k(&mut u_col, K_BITS);
    let u: Vector<{ PARAMS::K }> = negacyclic::column_to_vector::<{ PARAMS::K }>(&u_col);

    // ---- v = t^T r + e_2 + m'  (row-of-blocks matmul) ----------------------
    // Message: decompress the 1-bit ring under q=2^k => each bit -> bit << (k-1).
    let m_poly = negacyclic::decompress_ring_2k(&m.0, 1, K_BITS);

    let t_row = negacyclic::vector_to_row_negacyclic::<{ PARAMS::K }>(&t);
    let v_partial = t_row.dot(&r_col); // length-256 i128 vector

    // Add e_2 and the message into v, reduce mod 2^k.
    let mut v_ring = Ring::ZEROES_DEGREE255;
    let modulus: i128 = 1i128 << K_BITS;
    for c in 0..negacyclic::N {
        let mut x = v_partial[c];
        x += e_2.data[c] as i128;
        x += m_poly.data[c] as i128;
        let red = x.rem_euclid(modulus);
        debug_assert!(red >= 0 && red <= u32::MAX as i128);
        v_ring.data[c] = red as u32;
    }

    // ---- Compress u (D_U bits/coeff) and v (D_V bits/coeff) ----------------
    let u_compressed_ring = negacyclic::compress_vector_2k::<{ PARAMS::K }>(
        &u, PARAMS::D_U as u32, K_BITS,
    );
    let v_compressed_ring =
        negacyclic::compress_ring_2k(&v_ring, PARAMS::D_V as u32, K_BITS);

    // Wrap in the project's Compressed<D, T> newtype.
    let u_compressed = Compressed::<{ PARAMS::D_U }, Vector<{ PARAMS::K }>>(u_compressed_ring);
    let v_compressed = Compressed::<{ PARAMS::D_V }, Ring>(v_compressed_ring);

    (u_compressed, v_compressed)
}



















pub fn decrypt<PARAMS: MlKemParams>(dk_kpe: Vector<{PARAMS::K}>, c: Cyphertext<{PARAMS::K}, {PARAMS::D_U}, {PARAMS::D_V}>) -> Compressed<1, Ring> {
    //Decompress Cyphertext
    let u_compressed = c.0; // rA + e from the encryptor
    let v_compressed = c.1; // rt + e + m from the encryptor

    let u = u_compressed.decompress();

    let mut v = v_compressed.decompress();
    v.sub(&dk_kpe.inner_product(u.ntt()).inverse_ntt());

    Compressed::<1, Ring>::compress(v)
}


// ============================================================================
// decrypt for the Z_{2^k} branch.
//
// Drop-in replacement for the prime kpke::decrypt. Differences:
//   * NO NTT. s, u, v all in coefficient form.
//   * Decompress is exact under q=2^k: shift left by (k - D).
//   * s^T u via row-of-blocks negacyclic matmul (same trick encrypt used for
//     t^T r): one 256 x (256*K) matrix times the length-256*K u-column.
//   * Final 1-bit compress is the standard LWE decoder (rounds to nearest of
//     {0, q/2}).

pub fn decrypt_2k<PARAMS: MlKemParams>(
    dk_kpe: Vector<{ PARAMS::K }>,
    c: Cyphertext<{ PARAMS::K }, { PARAMS::D_U }, { PARAMS::D_V }>,
) -> Compressed<1, Ring> {
    use ndarray::Array1;

    let u_compressed = c.0; // Compressed<D_U, Vector<K>>
    let v_compressed = c.1; // Compressed<D_V, Ring>

    // ---- Decompress under q = 2^k (exact shift left) -----------------------
    let u: Vector<{ PARAMS::K }> = negacyclic::decompress_vector_2k::<{ PARAMS::K }>(
        &u_compressed.0,
        PARAMS::D_U as u32,
        K_BITS,
    );
    let v: Ring = negacyclic::decompress_ring_2k(&v_compressed.0, PARAMS::D_V as u32, K_BITS);

    // ---- Compute s^T u via row-of-blocks matmul ----------------------------
    let s_row = negacyclic::vector_to_row_negacyclic::<{ PARAMS::K }>(&dk_kpe);
    let u_col: Array1<i128> = negacyclic::vector_to_column::<{ PARAMS::K }>(&u);
    let su = s_row.dot(&u_col); // length-256 i128 vector

    // ---- v - s^T u  (mod 2^k) ----------------------------------------------
    let modulus: i128 = 1i128 << K_BITS;
    let mut diff = Ring::ZEROES_DEGREE255;
    for c in 0..negacyclic::N {
        let x = (v.data[c] as i128) - su[c];
        let red = x.rem_euclid(modulus);
        debug_assert!(red >= 0 && red <= u32::MAX as i128);
        diff.data[c] = red as u32;
    }

    // ---- Compress to 1 bit/coeff: LWE decoder ------------------------------
    let m_ring = negacyclic::compress_ring_2k(&diff, 1, K_BITS);
    Compressed::<1, Ring>(m_ring)
}