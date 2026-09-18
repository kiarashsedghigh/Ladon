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

The computational corresponding tables in the paper to the semi-honest version are Tables 3, 7, 10.

### Reproduce Table 3 (computation)

You should run:
```
cargo bench --bench encapsulation
```
and compare the `avg/encpas` with the first column of Table 3.

### Reproduce Table 7 (computation)

You should run:
```
cargo bench --bench kmstee
```
and compare the `sum` with the columns of Table 10 based on the parameter `t` for each security level.



### Table 10 (computation)
You should run:
```
cargo bench --bench keygen_sharing
```
and compare the `avg/op` with the columns of Table 10 based on the parameter `t` for each security level.


### Table 8 (Communication, not for this variant)

The following numbers for this benchmark are not in the paper. We have reported the communciation for the active variant in the paper. Check `active` folder for Table 8.

```
cargo bench --bench net
```