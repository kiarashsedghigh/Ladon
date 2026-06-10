//! Parameter sets for the Ladon active (SPDZ2k) branch.
//!
//! Q = 2^K_BITS is the PKE / ciphertext modulus. Power-of-two; q' = q in the
//! mod-switch step (identity) since q is already a power of 2.
pub const N: usize = 256;
pub const K_BITS: u32 = 30;
pub const Q: u32 = 1 << K_BITS;
pub const Q32: u32 = 1 << K_BITS;
pub const Q64: u64 = 1 << K_BITS;

use std::fmt;

/// Trait for adding parameter values to the Ladon parameter set structs.
/// Make functions generic over all parameter sets by using this trait as a bound.
pub trait MlKemParams {
    const K: usize;
    const ETA_1: usize;
    const ETA_2: usize;
    const D_U: usize;
    const D_V: usize;
}

/// Ladon parameter set at 128-bit computational security (paper Table 1).
/// Module rank K = 6.
pub struct Ladon128;
impl MlKemParams for Ladon128 {
    const K: usize = 6;
    const ETA_1: usize = 2;
    const ETA_2: usize = 2;
    const D_U: usize = 28;
    const D_V: usize = 23;
}

impl fmt::Debug for Ladon128 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Ladon128")
            .field("K", &<Ladon128 as MlKemParams>::K)
            .field("ETA_1", &<Ladon128 as MlKemParams>::ETA_1)
            .field("ETA_2", &<Ladon128 as MlKemParams>::ETA_2)
            .field("D_U", &<Ladon128 as MlKemParams>::D_U)
            .field("D_V", &<Ladon128 as MlKemParams>::D_V)
            .finish()
    }
}

/// Ladon parameter set at 256-bit computational security (paper Table 1).
/// Module rank K = 10 (vs 6 for Ladon128).
pub struct Ladon256;
impl MlKemParams for Ladon256 {
    const K: usize = 10;
    const ETA_1: usize = 2;
    const ETA_2: usize = 2;
    const D_U: usize = 28;
    const D_V: usize = 26;
}

impl fmt::Debug for Ladon256 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Ladon256")
            .field("K", &<Ladon256 as MlKemParams>::K)
            .field("ETA_1", &<Ladon256 as MlKemParams>::ETA_1)
            .field("ETA_2", &<Ladon256 as MlKemParams>::ETA_2)
            .field("D_U", &<Ladon256 as MlKemParams>::D_U)
            .field("D_V", &<Ladon256 as MlKemParams>::D_V)
            .finish()
    }
}

pub const ZETA: u32 = 3_073_009;