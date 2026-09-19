# Ladon - Active

## Set the Toolchain

Ladon requires **Rust nightly 1.94 or later** because it relies on the unstable `generic_const_exprs` feature.

Set the default Rust toolchain to nightly:

```bash
rustup default nightly
```


## Full Demo

Run the Ladon's demo where `(t, n) = (4, 5)` committee with a TEE (here a trusted system only) is used for performing a threshold decapsulation for both security levels (128 and 256-bits):

```bash
cargo run --release --bin ladon_demo
```

## Benchmarks and Tables

The computational benchmark results corresponding to the semi-honest version are reported in Tables 2, 6, and 9 of the paper.

### Table 2

You should run:
```
cargo bench --bench encapsulation
```
and compare the `avg/encpas` with the first column of Table 2 for Ladon.

### Table 6

You should run:
```
cargo bench --bench kmstee
```
and compare the `total` column with the columns of Table 6 for Ladon based on the parameter `t` for each security level.



### Table 9 
You should run:
```
cargo bench --bench keygen_sharing
```
and compare the `avg/op` with the columns of Table 9 for Ladon based on the parameter `t` for each security level.


### Table 8 (Communication)
You should run:
```
cargo bench --bench net
```
and compare the `total` with the columns of Table 8 for Ladon.