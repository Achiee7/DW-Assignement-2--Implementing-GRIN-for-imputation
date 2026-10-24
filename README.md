# GRIN reproduction on AQI-36

This project runs the AQI-36 out-of-sample experiment from Cini, Marisca and
Alippi, *Filling the G_ap_s: Multivariate Time Series Imputation by Graph Neural
Networks* (ICLR 2022).

The [Jupyter notebook](notebooks/grin_aqi36_reproduction.ipynb) contains the data
preparation, training, results and plots. The [report](report/assignment2_report.pdf)
has two pages summarising the paper and two pages describing the experiment.
The [complete report](report/assignment2_complete_report.pdf) also includes the
introduction, table of contents and highlighted source paper.

The CPU run uses **30 epochs, 512 windows per epoch and one seed**: 480 updates
in total. The paper uses up to 300 epochs and 5,120 windows per epoch, repeated
over five runs. Instructions for using that training budget are included below.

## Results

| Method | Source | MAE (µg/m³) | MSE |
|---|---|---:|---:|
| GRIN | Local run | 12.789 | 501.39 |
| GRIN | Paper, five-run mean | 12.08 ± 0.47 | 523.14 ± 57.17 |
| BRITS | Paper, five-run mean | 14.50 ± 0.35 | 662.36 ± 65.16 |
| Linear interpolation | Local baseline | 14.598 | 691.16 |
| Training station mean | Local baseline | 55.184 | 4721.55 |

The local GRIN MAE is 5.9% higher than the paper's mean and 12.4% lower than
linear interpolation. Training took 13.3 minutes, and epoch 18 had the lowest
validation error. MSE was lower than the published mean, but one run with less
training is not enough to conclude that GRIN performs better overall.

## Dataset and method

AQI-36 was chosen to compare the results with the paper on the same data.
It is the paper's smallest real sensor-network dataset. A new dataset would
test another application of GRIN, rather than repeat this experiment.

The notebook downloads the processed data: **8,759 hourly readings at 36
Beijing air-quality stations**. It uses the original evaluation mask, with
March, June, September and December reserved for testing. The splits contain
4,948 training, 564 validation and 2,788 test windows. Test error is calculated
on 20,434 withheld readings. Naturally missing readings are excluded from the
loss and evaluation.

The authors' model code is downloaded at a fixed commit and checked using
SHA-256. The training loop and data preparation are supplied by this project.
[PROTOCOL.md](PROTOCOL.md) lists the masks, splits, graph, loss and other settings.

## Installation

Use **Python 3.12**. The notebook was run on Windows with a CPU. The first run
downloads about 30 MB of data; installing PyTorch requires a larger download.

```text
git clone https://github.com/Achiee7/DW-Assignement-2--Implementing-GRIN-for-imputation.git
cd DW-Assignement-2--Implementing-GRIN-for-imputation
```

Create an environment in Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On Linux or macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Install the CPU version of PyTorch, the remaining packages and this project:

```text
python -m pip install --upgrade pip
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

If PowerShell cannot activate the environment, replace `python` with
`.\.venv\Scripts\python.exe` in the commands. Activation is optional.

`requirements.txt` pins the main packages. `requirements-lock.txt` lists the
complete tested Windows environment and can be used in its place for an exact
Windows installation. Use `requirements.txt` on other operating systems,
where some of the additional packages may differ.

For an NVIDIA GPU, use the [PyTorch installation selector](https://pytorch.org/get-started/locally/)
to install a PyTorch 2.14.1 CUDA wheel instead of the CPU wheel. Then install
the remaining packages as above. The saved results were obtained on CPU;
the GPU commands below have not been tested on the workstation yet.

## Open or rerun the notebook

The saved notebook can be read on GitHub without installing anything.
To execute it from the project directory:

```text
python scripts/run_notebook.py
```

This checks the saved settings, data and result hashes, then loads the existing
run and redraws the plots. If no completed run is available, it trains the
30-epoch model. The data are still needed to check the saved run and make the
plots. The executed notebook is saved to its original path.

For interactive use, open the project in VS Code, select the Python environment
and run the notebook cells in order. VS Code needs its Python and Jupyter
extensions. The command above works without a separate Jupyter server.

To train again:

```text
python scripts/run_notebook.py --profile bounded --seed 42 --device cpu --retrain
```

The new run is saved in a timestamped results folder, leaving the submitted
result in place. Its notebook is saved under `notebooks/runs/`.

## Use the paper's training budget

One seed on an NVIDIA GPU:

```text
python scripts/run_notebook.py --profile paper --seed 42 --device cuda
```

Use `--device cpu` for CPU training. To run five seeds in PowerShell:

```powershell
foreach ($seed in 42..46) {
    python scripts/run_notebook.py --profile paper --seed $seed --device cuda
}
```

In Bash:

```bash
for seed in 42 43 44 45 46; do
    python scripts/run_notebook.py --profile paper --seed "$seed" --device cuda
done
```

The authors did not publish their seeds, so these runs use seeds 42–46.
Results are saved under `results/paper_seed_<seed>/`. Repeating a completed
run loads its saved result. Use `--retrain` to start again. Saved results with
different settings or a different device are rejected rather than reused.

## Checks and saved files

Run the tests with:

```text
python -m pytest -q
```

They check the time splits, masks, loss and prediction averaging. The notebook
also checks that test targets stay hidden, predictions are finite and observed
values remain unchanged. The data splitter was compared with the authors'
implementation and returned the same indices.

Each results folder contains:

- `config.json`: training settings.
- `provenance.json`: data and source hashes, software versions and changes from the paper.
- `history.json`: training and validation results for each epoch.
- `best_model.pt`: checkpoint with the lowest validation error.
- `test_predictions.npz`: predictions, timestamps and evaluation mask.
- `metrics.json`: test errors and comparisons.
- `artifact_hashes.json`: checksums for the saved files.
- `figures/`: plots from the notebook.

Validation scores overlapping windows separately. For the final test score,
predictions for the same timestamp are averaged first, so each withheld reading
is scored once. This follows the authors' final evaluation. The local mean and
interpolation baselines were implemented separately from the paper's baselines.

## Files and sources

The notebook is in `notebooks/`, supporting code in `src/grin_reproduction/`,
and tests in `tests/`. `scripts/run_notebook.py` runs the notebook in the active
Python environment. The submitted results are in `results/bounded_seed_42/`.
The full method and checksums are listed in [PROTOCOL.md](PROTOCOL.md).

Raw data, Python environments and downloaded model files are excluded from Git.
Data and model downloads are checked against fixed hashes, including files
already in the cache. The first download requires internet access; verified
cached files can be reused offline.

GRIN was developed by **Andrea Cini, Ivan Marisca and Cesare Alippi**:

- [Paper, ICLR 2022](https://arxiv.org/abs/2108.00298v3)
- [Original model and experiment code](https://github.com/Graph-Machine-Learning-Group/grin)
- [Processed data mirror](https://huggingface.co/datasets/TorchSpatiotemporal/ProcessedDatasets/tree/v1.0.0/air)

The original model code is downloaded unchanged. This project supplies the data
preparation, training loop, notebook and tests. Generative AI assisted with the
supporting code, tests and explanations. The reported results were obtained by
executing the notebook.
