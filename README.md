![Logo](logo.png)
<h1 align="center">Ladon - Strengthening Confidential Computing by Decentralizing Trust
in the Key Management Service via a Threshold KEM

---

## Introduction for SP27 Evaluator PC

Ladon is implemented entirely in Rust and provides two variants: semi-honest and malicious security. In the paper, we report the performance results for these variants across several tables.

Each variant has a dedicated implementation directory in this project, and each directory includes a complete demo as well as benchmarking tools for reproducing the reported results.

The project also includes an additional `param_analyze` directory, which uses SageMath to evaluate the security of the selected parameter sets. As this analysis falls outside the scope of the IEEE S&P artifact evaluation objectives, it is not included in the main evaluation workflow.


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
