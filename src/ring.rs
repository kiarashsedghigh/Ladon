use crate::params;
use crate::util::*;

use std::sync::OnceLock;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum RingRepresentation {
    Degree255,
    NTT
}

#[derive(Clone, Debug)]
pub struct Ring {
    // Coefficients are now in Z_q with q = 8380417 (23-bit), so they no
    // longer fit in u16. Every coefficient is a u32; every *product* of two
    // coefficients is up to ~46 bits and MUST be computed in u64.
    pub data: [u32; 256],
    pub t: RingRepresentation
}


impl ToString for Ring {
    fn to_string(&self) -> String {
        let mut s = String::new();
        for i in 0..256 {
            s.push_str(format!("{:X}", self.data[i]).as_str());
            s.push(',');
        }
        s
    }
}

static ZETA_POWERS_MULT: OnceLock<[u32; 128]> = OnceLock::new();
static ZETA_POWERS_NTT: OnceLock<[u32; 128]> = OnceLock::new();

impl Ring {
    pub const ZEROES_NTT : Ring       = Ring { data: [0; 256], t: RingRepresentation::NTT };
    pub const ZEROES_DEGREE255 : Ring = Ring { data: [0; 256], t: RingRepresentation::Degree255 };

    fn get_zeta_powers_ntt() -> &'static [u32; 128] {
        ZETA_POWERS_NTT.get_or_init(|| {
            let mut zeta_powers = [0u32; 128];
            zeta_powers[0] = 1;
            for i in 1..128 {
                zeta_powers[i] = fastmodpow(params::ZETA, bitrev7(i as u8));
            };
            zeta_powers
        })
    }

    fn get_zeta_powers_mult() -> &'static [u32; 128] {
        ZETA_POWERS_MULT.get_or_init(|| {
            let mut zeta_powers = [0u32; 128];
            zeta_powers[0] = params::ZETA;
            for i in 1..128 {
                zeta_powers[i] = fastmodpow(params::ZETA, 2*bitrev7(i as u8) + 1);
            }
            zeta_powers
        })
    }

    pub fn scalar_mul(&mut self, value: u32) {
        let val = value as u64;

        for i in 0..256 {
            self.data[i] = ((self.data[i] as u64 * val) % params::Q64) as u32;
        }
    }

    pub fn add(&mut self, other: &Ring) -> &mut Self {
        // self.data[i] + other.data[i] < 2*Q < 2^24, safe in u32.
        for i in 0..256 {
            self.data[i] = (self.data[i] + other.data[i]) % params::Q;
        }
        self
    }

    pub fn sub(&mut self, other: &Ring) -> &mut Self {
        // self.data[i] + Q - other.data[i] < 2*Q < 2^24, safe in u32.
        for i in 0..256 {
            self.data[i] = (self.data[i] + params::Q - other.data[i]) % params::Q;
        }
        self
    }

    pub fn mult(&mut self, other: &Ring) -> &mut Self {
        if self.t != RingRepresentation::NTT || other.t != RingRepresentation::NTT {
            panic!("Multiplication requires NTT form");
        }

        let a = &mut self.data;
        let b = &other.data;

        let zeta_powers = Ring::get_zeta_powers_mult();

        for i in 0usize..128usize {
            let gamma = zeta_powers[i] as u64;

            // gamma * a[2i+1] * b[2i+1] would be ~69 bits and overflow u64,
            // so reduce the (a*b) product mod Q BEFORE multiplying by gamma.
            let prod = (a[2*i + 1] as u64 * b[2*i + 1] as u64) % params::Q64;

            let r_1: u64 = (a[2*i] as u64 * b[2*i] as u64 + gamma * prod) % params::Q64;
            let r_2: u64 = (a[2*i] as u64 * b[2*i + 1] as u64
                + a[2*i + 1] as u64 * b[2*i] as u64) % params::Q64;

            a[2*i] = r_1 as u32;
            a[2*i + 1] = r_2 as u32;
        }

        self
    }

    // In-Place, transforms ring to NTT form
    pub fn ntt(&mut self) -> &mut Self {
        const Q64: u64 = params::Q64;

        let zeta_powers = Ring::get_zeta_powers_ntt();

        if let Ring { data, t: RingRepresentation::Degree255 } = self {
            let mut k = 1;
            let mut len = 128;

            while len >= 2 {
                let mut start = 0;
                while start < 256 {
                    let zeta = zeta_powers[k] as u64;

                    k += 1;

                    for j in start..(start + len) {
                        // zeta * data[..] is ~46 bits, compute in u64.
                        let t = (zeta * data[j + len] as u64) % Q64;
                        data[j + len] = ((data[j] as u64 + Q64 - t) % Q64) as u32;
                        data[j] = ((data[j] as u64 + t) % Q64) as u32;
                    }

                    start += 2 * len;
                }

                len /= 2;
            }

            self.t = RingRepresentation::NTT;

            self
        } else {
            panic!("Ring is already in NTT form");
        }
    }

    pub fn inverse_ntt(&mut self) -> &mut Self {
        const Q64: u64 = params::Q64;

        let zeta_powers = Ring::get_zeta_powers_ntt();

        if let Ring {  data, t: RingRepresentation::NTT } = self {
            let mut k = 127;
            let mut len = 2;
            while len <= 128 {
                let mut start = 0;
                while start < 256 {
                    let zeta = zeta_powers[k] as u64;

                    k -= 1;

                    for j in start..(start + len) {
                        let t = data[j];
                        data[j] = (data[j + len] + t) % params::Q;
                        // (Q + data[j+len] - t) < 2Q < 2^24; times zeta (~23 bit) ~ 46 bits -> u64.
                        data[j + len] = ((zeta * (params::Q + data[j + len] - t) as u64) % Q64) as u32;
                    }

                    start += 2 * len;
                }

                len *= 2;
            }

            // 128^-1 mod 8380417 (this NTT normalizes by 128, matching the
            // degree-2 base case; 3303 = 128^-1 mod 3329 in the original).
            self.scalar_mul(682286673);

            self.t = RingRepresentation::Degree255;

            self
        } else {
            panic!("Ring is already in Degree255 form");
        }
    }
}

impl PartialEq for Ring {
    fn eq(&self, other: &Self) -> bool {
        for i in 0..256 {
            if self.data[i] != other.data[i] {
                return false;
            }
        }
        true
    }
}

//Tuple that is either all ring or all ntt, k is length
#[derive(Clone, Debug)]
pub struct Vector<const K: usize> {
    pub data: [Ring; K]
}

impl<const K: usize> Vector<K> {
    pub fn new_ntt() -> Vector<K> {
        Vector {
            data: core::array::from_fn(|_| Ring::ZEROES_NTT)
        }
    }

    pub fn new_degree255() -> Vector<K> {
        Vector {
            data: core::array::from_fn(|_| Ring::ZEROES_DEGREE255)
        }
    }

    pub fn add(&mut self, other: &Vector<K>) -> &mut Self {
        for i in 0..K {
            self.data[i].add(&other.data[i]);
        }
        self
    }

    pub fn inner_product(mut self, other: Vector<K>) -> Ring {
        let mut result = match self.data[0].t {
            RingRepresentation::Degree255 => Ring::ZEROES_DEGREE255,
            RingRepresentation::NTT => Ring::ZEROES_NTT
        };

        for i in 0..K {
            result.add(
                self.data[i].mult( &other.data[i] )
            );
        }
        result
    }

    // NTT and NTT^-1 are in place and chainable
    pub fn ntt(mut self) -> Self {
        for i in 0..K {
            self.data[i].ntt();
        }
        self
    }

    pub fn inverse_ntt(mut self) -> Self {
        for i in 0..K {
            self.data[i].inverse_ntt();
        }
        self
    }
}

impl<const K: usize> PartialEq for Vector<K> {
    fn eq(&self, other: &Self) -> bool {
        for i in 0..K {
            if self.data[i] != other.data[i] {
                return false;
            }
        }
        true
    }
}

// Represent a compressed ring or vector as its own type

#[derive(PartialEq, Clone, Debug)]
pub struct Compressed<const D: usize, T> (pub T);

pub trait CompressionConstants<const D: usize> {
    const POW_HALF: u32;
    const BITMASK: u32;
}

impl<const D: usize, T> CompressionConstants<D> for Compressed<D, T> {
    const POW_HALF: u32 = 1 << (D - 1);
    // D can now exceed 16, so the mask must be u32.
    const BITMASK: u32 = (1 << D) - 1;
}

const Q_HALF: u64 = params::Q64 / 2;

impl<const D: usize> Compressed<D, Ring> {
    pub fn compress(mut ring: Ring) -> Compressed<D, Ring> {
        for i in 0..256 {
            // ((x << D) + Q/2) / Q : with x ~23 bit and D up to ~13, the
            // shifted value is ~36 bits -> compute in u64.
            ring.data[i] = ((((ring.data[i] as u64) << D) + Q_HALF) / params::Q64) as u32
                & Compressed::<D, Ring>::BITMASK;
        }

        Compressed (ring)
    }

    pub fn decompress(self) -> Ring {
        let Compressed (mut ring) = self;

        for i in 0..256 {
            // x * Q is ~ (D bits + 23 bits) -> u64.
            ring.data[i] = (((ring.data[i] as u64) * params::Q64
                + Compressed::<D, Ring>::POW_HALF as u64) >> D) as u32;
        }

        ring
    }
}

impl<const D: usize, const K: usize> Compressed<D, Vector<K>> {
    pub fn compress(mut vector: Vector<K>) -> Compressed<D, Vector<K>> {
        for i in 0..K {
            for j in 0..256 {
                vector.data[i].data[j] = ((((vector.data[i].data[j] as u64) << D) + Q_HALF) / params::Q64)
                    as u32 & Compressed::<D, Vector<K>>::BITMASK;
            }
        }

        Compressed (vector)
    }

    pub fn decompress(self) -> Vector<K> {
        let Compressed (mut vector) = self;

        for i in 0..K {
            for j in 0..256 {
                vector.data[i].data[j] = (((vector.data[i].data[j] as u64) * params::Q64
                    + Compressed::<D, Vector<K>>::POW_HALF as u64) >> D) as u32;
            }
        }

        vector
    }
}


#[derive(Clone, Debug)]
pub struct Matrix<const K: usize> {
    pub data: [[Ring; K]; K]
}

impl<const K: usize> Matrix<K> {

    pub fn new_ntt() -> Matrix<K> {
        Matrix {
            data: core::array::from_fn(|_| core::array::from_fn(|_| Ring::ZEROES_NTT))
        }
    }

    // These matrix operations are only ever done once, so they can be consuming on the matrix
    pub fn right_vector_multiply(mut self, vector: &Vector<K>) -> Vector<K> {
        debug_assert_eq!(self.data[0][0].t, RingRepresentation::NTT);
        debug_assert_eq!(vector.data[0].t, RingRepresentation::NTT);

        let mut result = Vector::new_ntt();

        for i in 0..K {
            for j in 0..K {
                result.data[i].add(self.data[i][j].mult(&vector.data[j]));
            }
        }
        result
    }

    pub fn left_vector_multiply(mut self, vector: &Vector<K>) -> Vector<K> {
        debug_assert_eq!(self.data[0][0].t, RingRepresentation::NTT);
        debug_assert_eq!(vector.data[0].t, RingRepresentation::NTT);

        let mut result = Vector::new_ntt();
        for i in 0..K {
            for j in 0..K {
                result.data[i].add(self.data[j][i].mult(&vector.data[j]));
            }
        }
        result
    }
}