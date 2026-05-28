use crate::params::Q64;

pub fn bitrev7(x: u8) -> u8 {
    let mut x = x;
    let mut y = 0;
    for _ in 0..7 {
        y = (y << 1) | (x & 1);
        x >>= 1;
    }
    y
}

pub fn fastmodpow(base: u32, exp: u8) -> u32 {
    // q is now ~2^23, so base*base is ~2^46 and overflows u32.
    // All multiplications are done in u64 and reduced mod q.
    let mut base = base as u64;
    let mut exp = exp as u64;
    let mut result: u64 = 1;

    while exp > 0 {
        if exp % 2 == 1 {
            result = (result * base) % Q64;
        }
        exp = exp >> 1;
        base = (base * base) % Q64;
    }
    result as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::ZETA;

    #[test]
    fn test_bitrev7() {
        assert_eq!(127, bitrev7(127));
        assert_eq!(1, bitrev7(64));
        assert_eq!(64, bitrev7(1));
        assert_eq!(0, bitrev7(0));
    }

    #[test]
    fn test_fastmodpow() {
        // Expected values recomputed for ZETA = 1753 mod 8380417.
        assert_eq!(fastmodpow(ZETA, 1), ZETA);
        assert_eq!(fastmodpow(ZETA, 2), ZETA * ZETA); // 1753^2 = 3073009 < q, so no reduction
        assert_eq!(fastmodpow(ZETA, 10), 1528066);
        assert_eq!(fastmodpow(ZETA, 100), 3704823);
        assert_eq!(fastmodpow(ZETA, 255), 7648983);
    }
}