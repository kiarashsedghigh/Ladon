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
