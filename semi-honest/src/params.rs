pub const N : usize = 256;
pub const Q: u32 = 687659009;
pub const Q32: u32 = 687659009;
pub const Q64: u64 = 687659009;
pub const ZETA: u32 = 174453379;

use std::fmt;


/// Trait for adding parameter values to the 3 parameter set structs.
///
/// You can make new functions generic over all 3 parameter sets by using this trait as a bound.
pub trait MlKemParams {
    const K: usize;
    const ETA_1: usize;
    const ETA_2: usize;
    const D_U: usize;
    const D_V: usize;
}

/// Ladon parameter set at 128-bit computational security (paper Table 1).
/// Same numerical parameters as `MlKem512` in this codebase; kept as a
/// separate struct so benches and demos can reference the Ladon naming
/// directly without coupling to the ML-KEM-512 label.
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
/// Differs from Ladon128 only in the module rank K (10 vs 6).
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

