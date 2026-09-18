# Ladon - Semi-Honest

## Set the Toolchain

Ladon requires **Rust nightly 1.94 or later** because it uses the unstable `generic_const_exprs` feature.

Set the default Rust toolchain to the latest nightly version:

```bash
rustup default nightly
```


## Full Demo

Run Ladon's full demo using a (t, n) = (4, 9) committee and a TEE (implemented here as a trusted local system) to perform threshold decapsulation at both supported security levels: **128-bit** and **256-bit**. 

```bash
cargo run --release --bin ladon_demo
```


## Benchmarks and Tables

The computational benchmark results corresponding to the semi-honest version are reported in Tables 3, 7, and 10 of the paper.

Table 5 for semi-honest reports the analytical communication cost and therefore does not correspond to any executable code.


### Table 3 

You should run:
```
cargo bench --bench encapsulation
```
and compare the `avg/encpas` with the first column of Table 3.

### Table 7

You should run:
```
cargo bench --bench kmstee
```
and compare the `total` column with the columns of Table 7 based on the parameter `t` for each security level.


### Table 10
You should run:
```
cargo bench --bench keygen_sharing
```
and compare the `avg/op` with the columns of Table 10 based on the parameter `t` for each security level.
