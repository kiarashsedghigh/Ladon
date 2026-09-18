# Ladon - Semi-Honest

## Set the Toolchain

Ladon requires **Rust nightly 1.94 or later** because it uses the unstable `generic_const_exprs` feature.

Set the default Rust toolchain to nightly:

```bash
rustup default nightly
```

## Full Demo

Run the Ladon's demo where `(t, n) = (4, 9)` committee with a TEE (here a trusted system only) is used for performing a threshold decapsulation for both security levels (128 and 256-bits):

```bash
cargo run --release --bin ladon_demo
```

## Benchmarks and Tables

The computational corresponding tables in the paper to the semi-honest version are Tables 2, 6, 9.

### Reproduce Table 2 (computation)

You should run:
```
cargo bench --bench encapsulation
```
and compare the `avg/encpas` with the first column of Table 2.

### Reproduce Table 6 (computation)

You should run:
```
cargo bench --bench kmstee
```
and compare the `sum` with the columns of Table 6 based on the parameter `t` for each security level.



### Table 9 (computation)
You should run:
```
cargo bench --bench keygen_sharing
```
and compare the `avg/op` with the columns of Table 9 based on the parameter `t` for each security level.


### Table 8 (Communication)
You should run:
```
cargo bench --bench net
```