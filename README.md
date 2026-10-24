# GRIN: AQI-36 reproduction for Data Wrangling

This submission evaluates GRIN on the original **AQI-36 out-of-sample imputation
benchmark** from Cini, Marisca and Alippi, *Filling the G_ap_s: Multivariate Time
Series Imputation by Graph Neural Networks* (ICLR 2022).

The main artifact is an executable
[Jupyter notebook](notebooks/grin_aqi36_reproduction.ipynb). It prepares the data,
explains the masks and graph, trains or verifies a saved model, plots results,
and compares measured error with the paper. The report describes the procedure,
parameter settings and differences; the notebook performs the implementation.

The [four-page report PDF](report/assignment2_report.pdf) contains the two-page
paper summary followed by the two-page reproduction report. The highlighted
source paper is supplied separately with the Word report, rather than copied
into this public code repository.

**Scope:** the submitted run uses one seed, 30 epochs and 512 sampled windows per
epoch: at most **480 updates**. The paper permits 300 epochs and 5,120 windows per
epoch, up to **48,000 updates per run**, and reports five runs. This is a bounded
reproduction attempt with the original architecture and evaluation protocol,
not a claim to reproduce the complete published experiment. The full training
budget is available as an explicit option below.

## Measured result

| Method | Result source | MAE (µg/m³) | MSE |
|---|---|---:|---:|
| GRIN | Local 30-epoch single-seed run | 12.789 | 501.39 |
| GRIN | Paper, five-run mean | 12.08 ± 0.47 | 523.14 ± 57.17 |
| BRITS | Paper, five-run mean | 14.50 ± 0.35 | 662.36 ± 65.16 |
| Linear interpolation | Local control | 14.598 | 691.16 |
| Training station mean | Local control | 55.184 | 4721.55 |

The local GRIN MAE is 5.9% higher than the published mean,
and 12.4% lower than the local interpolation control.
The run completed 480 updates in 13.3 minutes and selected epoch 18 by
validation MAE. Its lower MSE does not establish overall superiority: this is
one reduced-budget seed, while the paper reports five runs.

## Which dataset is used?

The assignment asks to reproduce a main experiment and compare the result with
the paper. The original dataset gives a meaningful comparison; using a new
dataset alone would be an additional application rather than reproduction of
that experiment. AQI-36 is the smallest real sensor-network benchmark in GRIN.

The notebook downloads the authors' processed data mirror automatically:
**8,759 hourly readings × 36 Beijing air-quality stations**, with the original
fixed evaluation mask and geographical graph. Test months are March, June,
September and December. There are 4,948 training, 564 validation and 2,788 test
windows; final testing scores 20,434 withheld readings. Natural missing values
are never treated as observed targets. See [PROTOCOL.md](PROTOCOL.md) for exact
mask, split, scaling, graph, optimization and scoring details.

## Installation

Use **Python 3.12**. The tested environment is Windows CPU; Linux users should
also use Python 3.12. Approximately 30 MB of processed data and several original
model files are downloaded on first use. Installing PyTorch requires additional
disk space and download time.

Clone the public submission:

```text
git clone https://github.com/Achiee7/dw-grin-reproduction.git
cd dw-grin-reproduction
```

On Windows PowerShell, create and activate an environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On Linux/macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Then install the tested CPU dependencies and local package:

```text
python -m pip install --upgrade pip
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

If PowerShell activation is unavailable, use `.\.venv\Scripts\python.exe` in
place of `python` in each command; activation is not required for execution.

`requirements.txt` pins the direct dependencies used by the notebook.
`requirements-lock.txt` records the complete tested Windows CPU environment.
For an exact Windows environment, install that lock file instead of
`requirements.txt` after installing CPU PyTorch. On other operating systems,
use `requirements.txt`: some transitive packages in the complete lock are
platform dependent. Neither file is a claim of identical floating-point results
across hardware and software platforms.

For an NVIDIA GPU, install the corresponding PyTorch 2.14.1 CUDA wheel for the
machine using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/)
**instead of the CPU-wheel command**. Then install the remaining requirements
and package normally. Use `--device cuda` when running the notebook. The checked
submission was executed on CPU; a CUDA run is an additional run and requires a
compatible GPU and driver.

## Review the submitted result

Open [the executed notebook](notebooks/grin_aqi36_reproduction.ipynb) directly on
GitHub to inspect the explanations, recorded output and figures without running
anything. Machine-readable results live in
[`results/bounded_seed_42/metrics.json`](results/bounded_seed_42/metrics.json).
The published comparison is **out-of-sample AQI-36 GRIN MAE 12.08 ± 0.47 µg/m³**;
the in-sample score 10.51 is a different experiment. Actual local measurements,
including their differences from the paper, are recorded in the notebook and
metrics file. No published value is substituted for a measured result.

From the cloned project directory, this command executes every notebook cell
using the active Python environment:

```text
python scripts/run_notebook.py
```

When the completed submission artifacts are present, this verifies the saved
configuration, data fingerprint, source hashes and artifact checksums, then
loads the recorded result and regenerates the notebook's comparisons and plots.
It does **not** train a fresh model by default when a valid completed run exists.
The initial data download is still needed to verify the data and generate plots.
If no completed result exists, the same command trains the bounded experiment.
The executed notebook is saved back to its original path.

For interactive use, open the project folder in VS Code, install its Python and
Jupyter extensions, open the notebook, and select the environment created above.
Run all cells in order. A separate JupyterLab or browser-notebook server is not
required by the headless command or the VS Code workflow.

## Train a fresh model

Re-run the submitted bounded configuration, including a fresh optimization:

```text
python scripts/run_notebook.py --profile bounded --seed 42 --device cpu --retrain
```

If a completed result already exists, retraining writes to a timestamped result
directory so the submitted measurement is retained. The executed alternative
notebook is written beneath `notebooks/runs/`. Model selection uses validation
data only. Test metrics are calculated after loading the best checkpoint.

Run one seed with the paper's full training budget:

```text
python scripts/run_notebook.py --profile paper --seed 42 --device cuda
```

Use `--device cpu` for CPU execution. Full-budget training takes substantially
longer than the submitted bounded run. One full-budget seed still does not
reproduce the paper's five-run mean and variability. To execute five seeds in
PowerShell:

```powershell
foreach ($seed in 42..46) {
    python scripts/run_notebook.py --profile paper --seed $seed --device cuda
}
```

Equivalent Bash:

```bash
for seed in 42 43 44 45 46; do
    python scripts/run_notebook.py --profile paper --seed "$seed" --device cuda
done
```

These are explicit local seeds, not a claim to know the authors' unreleased
seed values. Each seed writes `results/paper_seed_<seed>/`. Repeating a completed
seed loads its verified result unless `--retrain` is supplied. Results saved
with another device or configuration are intentionally rejected as a cache
mismatch; use `--retrain` to obtain a fresh matching run.

The complete run configuration is in each result directory's `config.json`.
The published architecture has hidden/feedforward dimensions 64, node embedding
dimension 8, one recurrent layer, encoder diffusion order 2, decoder order 1,
and an MLP bidirectional merge. Both profiles use Adam at 0.001, cosine decay
to 0.0001, batch size 32, 5% additional training withholding, gradient norm
clipping at 5, and validation patience 40. The bounded profile compresses the
cosine schedule to 30 epochs. [PROTOCOL.md](PROTOCOL.md) lists all settings,
including the five-term masked MAE objective and known implementation changes.

## Validation and saved evidence

Run the automated checks:

```text
python -m pytest -q
```

The checks cover temporal separation, mask behavior, the loss and overlapping
prediction aggregation. Notebook checks also verify hidden test targets,
finite predictions, preservation of observed values and saved-artifact
integrity. The independent data splitter was compared against every index
returned by the pinned authors' implementation.

Each completed run contains:

- `config.json`: every selected training parameter.
- `provenance.json`: data/source fingerprints, software versions and deviations.
- `history.json`: training and validation results by epoch.
- `best_model.pt`: validation-selected checkpoint.
- `test_predictions.npz`: aggregated predictions, timestamps and evaluation mask.
- `metrics.json`: measured metrics, paper comparison and local control baselines.
- `artifact_hashes.json`: SHA256 checksums of the saved run artifacts.
- `figures/`: figures generated by the notebook.

Validation selection uses **window-weighted MAE**. Final test scoring first
averages overlapping predictions by timestamp and then scores every withheld
reading once, matching the original final evaluation. These metrics are saved
separately and should not be interchanged. The locally implemented station-mean
and interpolation baselines are controls; they are not presented as reproductions
of the paper's baseline implementations.

## Project files

```text
notebooks/grin_aqi36_reproduction.ipynb  Executable submission and recorded output
scripts/run_notebook.py                Execute in the selected Python environment
src/grin_reproduction/data.py          Verified dataset and independent preprocessing
src/grin_reproduction/upstream.py      Fetch/import unchanged pinned GRIN model
src/grin_reproduction/experiment.py    Training, selection, evaluation and integrity
src/grin_reproduction/upstream_manifest.json  Per-file source checksums
results/bounded_seed_42/               Submitted measured run and checkpoint
tests/                                Focused verification checks
PROTOCOL.md                           Full methodological details and provenance
requirements.txt                      Pinned direct dependencies
requirements-lock.txt                 Complete tested Windows CPU environment
```

Raw data, local environments and downloaded upstream sources are excluded from
Git. Download verification is mandatory: the archive SHA256 is
`95d3c5cab14fc007f43c57b745a93e5f88201cbb92a755644ca9777848343317`;
the extracted AQI-36 file SHA256 is
`aedcb5080287fba05e99cd4b692300968b9c33113117db9fa37f2aa955a1091d`.
The model loader retrieves selected files from immutable upstream commit
`4a28afbb092600b6e6abeeabaaf67e87dbd1ed6e` and checks every file against
`upstream_manifest.json`, including already cached files. An internet connection
is needed for the first data/source downloads; keep verified caches for later
offline use.

## Attribution and assistance

GRIN is the work of **Andrea Cini, Ivan Marisca and Cesare Alippi**:

- [Paper, ICLR 2022](https://arxiv.org/abs/2108.00298v3)
- [Original model and experiment code](https://github.com/Graph-Machine-Learning-Group/grin)
- [Versioned processed data mirror](https://huggingface.co/datasets/TorchSpatiotemporal/ProcessedDatasets/tree/v1.0.0/air)

The model is the authors' implementation, downloaded unchanged at runtime;
it is not presented as an original implementation by this submission. The
project independently supplies data preparation, a modern training/evaluation
harness, notebook, checks and documentation. Original upstream source and raw
data are not redistributed here. Attribution does not transfer their ownership.

Generative AI tools assisted with inspecting the paper/code, implementing the
experiment harness and checks, and drafting notebook/report explanations and
documentation. The run artifacts record what was actually executed. Limits of
the training budget and differences from the original experiment remain
explicit rather than being hidden behind the published results.
