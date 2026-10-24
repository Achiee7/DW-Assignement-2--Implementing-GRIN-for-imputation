# GRIN reproduction

This project reproduces the AQI-36 out-of-sample imputation experiment from
Cini, Marisca and Alippi, *Filling the G_ap_s: Multivariate Time Series Imputation
by Graph Neural Networks*, ICLR 2022.

The executable Jupyter notebook is the main submission artifact. The report
explains the method, experiment, parameter settings and measured differences.

## Status

The notebook and experiment are being validated. Measured results and exact
run instructions will be added before the submission is finalised.

## Sources

- Paper: https://arxiv.org/abs/2108.00298v3
- Authors' code: https://github.com/Graph-Machine-Learning-Group/grin
- Pinned source commit: `4a28afbb092600b6e6abeeabaaf67e87dbd1ed6e`
- Processed data: https://huggingface.co/datasets/TorchSpatiotemporal/ProcessedDatasets/tree/v1.0.0/air

The source loader downloads and verifies the original model at runtime. Source
code, data and checkpoints remain attributable to their respective authors.
