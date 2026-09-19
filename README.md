![Logo](logo.png)
<h1 align="center">Ladon - Strengthening Confidential Computing by Decentralizing Trust
in the Key Management Service via a Threshold KEM

---

## Introduction for SP27 Evaluator PCs

Ladon is implemented entirely in Rust and provides two variants: semi-honest and malicious security. In the paper, we report the performance results for these variants across several tables.

Each variant has a dedicated implementation directory in this project, and each directory includes a complete demo as well as benchmarking tools for reproducing the reported results.


## Software Requirements

Ladon requires **Rust nightly 1.94 or later**. The latest version of Rust can be installed by following the instructions on the official Rust website:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```



## Evaluation 

To run either variant, first `cd` into the directory corresponding to the desired security model. Then, refer to the `README.md` file in that directory for instructions on running the demos and benchmarks, as well as for mapping the benchmark results to the corresponding tables in the paper.

1. **Semi-honest variant.**
   ```bash
   cd semi-honest
   ```
   Then follow the instructions in `semi-honest/README.md`.

2. **Active variant.**
   ```bash
   cd active
   ```
   Then follow the instructions in `active/README.md`.
