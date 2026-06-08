use crate::ring::*;
use crate::params;
use crate::crypt;

use bitvec::prelude::*;

pub fn sample_ntt(mut xof_stream: crypt::XOF) -> Ring {
    let mut ring: Ring = Ring::ZEROES_NTT;
    let mut three_bytes = [0u8; 3];

    let mut j = 0;
    while j < 256 {
        // q = 8380417 is a 23-bit prime, so 12-bit samples can no longer
        // cover Z_q. Instead, each 3 bytes (24 bits) yield ONE candidate:
        // mask off the top bit to get a 23-bit value, then reject if >= q.
        // This is the ML-DSA-style rejection sampler; acceptance ~99.9%.
        xof_stream.get_3_bytes(&mut three_bytes);

        let b0 = three_bytes[0] as u32;
        let b1 = three_bytes[1] as u32;
        let b2 = three_bytes[2] as u32;

        // 23-bit uniform candidate (top bit of b2 cleared).
        let d = b0 | (b1 << 8) | ((b2 & 0x7F) << 16);

        if d < params::Q {
            ring.data[j] = d;
            j += 1;
        }
    }

    ring
}

pub fn sample_poly_cbd<const ETA: usize>(byte_array: [u8; 64*ETA]) -> Ring
{
    let b = byte_array.view_bits::<Lsb0>();
    let mut f: Ring = Ring::ZEROES_DEGREE255;

    for i in 0..256 {
        // x, y are small (<= ETA), but coefficients are u32 now and
        // params::Q is u32, so do the modular combine in u32.
        let mut x = 0u32;
        let mut y = 0u32;

        for j in 0..ETA {
            x += b[i*2*ETA + j] as u32;
            y += b[i*2*ETA + j + ETA] as u32;
        }

        f.data[i] = (x + params::Q - y) % params::Q;
    }

    f
}
