# Experiment protocol and provenance

This project evaluates the authors' GRIN architecture on the original **AQI-36
out-of-sample imputation benchmark**. The executed notebook uses a smaller
training budget and one seed. It is a bounded reproduction attempt, not a
reproduction of the complete five-run paper experiment. The original data,
fixed evaluation mask, graph, architecture, objective and scoring protocol are
retained. No test scores are used for model selection.

## Source versions

- Paper: Cini, Marisca and Alippi (ICLR 2022),
  [Filling the G_ap_s](https://arxiv.org/abs/2108.00298), Sections 4–5 and Appendix A.
- Original implementation: commit
  `4a28afbb092600b6e6abeeabaaf67e87dbd1ed6e` of
  [Graph-Machine-Learning-Group/grin](https://github.com/Graph-Machine-Learning-Group/grin/tree/4a28afbb092600b6e6abeeabaaf67e87dbd1ed6e).
- Authors' experiment settings:
  [air36.yaml](https://github.com/Graph-Machine-Learning-Group/grin/blob/4a28afbb092600b6e6abeeabaaf67e87dbd1ed6e/config/grin/air36.yaml),
  [run_imputation.py](https://github.com/Graph-Machine-Learning-Group/grin/blob/4a28afbb092600b6e6abeeabaaf67e87dbd1ed6e/scripts/run_imputation.py),
  [AirQuality reader and splitter](https://github.com/Graph-Machine-Learning-Group/grin/blob/4a28afbb092600b6e6abeeabaaf67e87dbd1ed6e/lib/datasets/air_quality.py),
  [scaler](https://github.com/Graph-Machine-Learning-Group/grin/blob/4a28afbb092600b6e6abeeabaaf67e87dbd1ed6e/lib/data/preprocessing/scalers.py),
  [GraphFiller objective](https://github.com/Graph-Machine-Learning-Group/grin/blob/4a28afbb092600b6e6abeeabaaf67e87dbd1ed6e/lib/fillers/graphfiller.py).
- Data mirror: the authors' successor library
  [Torch Spatiotemporal](https://github.com/TorchSpatiotemporal/tsl/blob/main/tsl/datasets/air_quality.py)
  identifies the versioned processed-data download below. The archive's checksum
  fixes its contents independently of future changes to that library.

## Data download and checksums

The notebook downloads
[air_quality.zip, version v1.0.0](https://huggingface.co/datasets/TorchSpatiotemporal/ProcessedDatasets/resolve/v1.0.0/air/air_quality.zip)
into `data/raw/`, checks its SHA256, and extracts only `small36.h5`.
Dataset files are downloaded rather than redistributed in this repository.

| File | SHA256 |
|---|---|
| `air_quality.zip` | `95d3c5cab14fc007f43c57b745a93e5f88201cbb92a755644ca9777848343317` |
| `small36.h5` | `aedcb5080287fba05e99cd4b692300968b9c33113117db9fa37f2aa955a1091d` |

The released file contains **8,759 hourly observations from 36 stations**, from
2014-05-01 01:00 to 2015-04-30 23:00. The paper describes a year of measurements;
the missing first midnight is not inserted. The file supplies PM2.5 readings,
station coordinates and the fixed evaluation mask. It contains 41,771 naturally
missing entries and 35,737 designated evaluation targets across the whole year.

## Masks and preprocessing

- `observed_mask` is true only where a real reading exists.
- `eval_mask` is the file's fixed withholding pattern. Every selected target
  has a real reading, so imputation error can be calculated. It is not a newly
  drawn independent 25% mask.
- Initially, `input_mask = observed_mask & ~eval_mask`. Missing and withheld
  values are replaced with zero **after standardization** before entering GRIN.
  A zero therefore represents a placeholder; the mask distinguishes it from
  an observed reading equal to the training mean.
- As in the authors' reader, naturally missing numeric placeholders are filled
  by hour within week, then progressively broader calendar groups and finally
  forward/backward fill. These entries remain masked from inputs and loss;
  they never become ground truth. Modern pandas' string `mean` implementation
  replaces the original NumPy callable and differs by at most 0.0000153 in these
  masked float32 placeholders in the verification run.
- One global mean and population standard deviation are fitted using visible
  readings in training-covered timestamps only. This follows the original
  ordering: fit the scaler while fixed evaluation targets are excluded, then
  restore those targets as visible readings **within training intervals only**.
  Validation and test targets remain hidden. The resulting mean is
  `90.6363754272461`, standard deviation `83.54097747802734`.

## Temporal splits and graph

Windows contain 36 consecutive hours with stride 1; input and reconstruction
target cover the same window. A test window must start and finish in March,
June, September or December. Validation windows are placed before each test
month using the released splitter's 0.1 validation fraction and 36-hour offset.
Training windows overlapping any validation observations are removed. No
observation occurs in two splits. The independent splitter was checked against
all indices returned by the pinned original implementation.

| Split | Windows | Final test scoring |
|---|---:|---|
| Training | 4,948 | Not scored as test data |
| Validation | 564 | Used for checkpoint selection |
| Testing | 2,788 | 20,434 withheld readings across 2,928 hours |

Geographical distances use the haversine formula and Earth radius 6,371.0088 km.
Graph weights are `exp(-(distance / theta)**2)`, where `theta` is the standard
deviation of all pairwise distances: `26.677876821579837` km. Weights below 0.1
are set to zero, as is the diagonal. The graph has 654 nonzero directed entries
(it is symmetric). GRIN constructs forward and reverse normalized diffusion
supports from this graph.

## Architecture, optimization and executed budget

The original model files are fetched at the pinned commit into a local cache.
`upstream_manifest.json` lists the verified SHA256 of every downloaded file.
The source is imported unchanged; legacy package initializers are bypassed.

| Parameter | Executed notebook | Paper configuration |
|---|---|---|
| Hidden / feedforward / node embedding sizes | 64 / 64 / 8 | Same |
| Recurrent layers; encoder diffusion order; decoder order | 1; 2; 1 | Same |
| Bidirectional merge | MLP | Same |
| Feedforward dropout; layer normalization; global attention | 0; disabled; disabled | Same |
| Batch size | 32 | 32 |
| Epoch limit | 30 | 300 |
| Training windows sampled per epoch, with replacement | 512 | 5,120 |
| Optimizer / weight decay | Adam / 0 | Same |
| Initial learning rate / cosine minimum | 0.001 / 0.0001 | Same |
| Cosine schedule duration | 30 epochs | 300 epochs |
| Gradient clipping | Global norm, maximum 5 | Same |
| Additional random training withholding | 5% | Same |
| Early stopping | Patience 40; minimum validation window MAE | Same |
| Seed / number of runs | 42 / 1 | Five runs; exact seeds not supplied |
| Execution | CPU; one PyTorch thread; deterministic algorithms | Original software/hardware differs |

Each training batch additionally hides 5% of visible input readings. The loss
is the **sum of five equally weighted masked MAEs**: final bidirectional output
and the two directional first-stage and second-stage readouts. Targets are
standardized. Every term is scored on the original pre-whitening visible
training mask, following `GraphFiller`; naturally missing entries are excluded.

The notebook's maximum budget is 480 optimizer updates, versus up to 48,000 per
paper-configured run. `Config.paper(seed=42, device="cpu")` exposes the published
300 × 5,120 budget. Run separately for multiple seeds to estimate variability.
The compressed cosine schedule and NumPy window sampler are documented
differences: the original uses PyTorch's replacement sampler. A fixed local seed
therefore provides local reproducibility, not the authors' identical batch order.

## Selection, scoring and comparison

After every epoch, validation MAE is measured in original PM2.5 units over all
withheld entries in all validation windows. Overlapping timestamps occur more
than once in this **window-weighted** selection metric, as in the original
Lightning validation procedure. Only the best validation checkpoint is tested.

For final test metrics, first average overlapping window predictions for each
timestamp and sensor. Then score each withheld test reading once. This
**timestamp-aggregated** metric matches the final results printed by the
authors' script. MAE and MSE use the raw units; MRE is
`sum(abs(error)) / sum(abs(ground_truth))`, reported as a ratio in JSON and as a
percentage when multiplied by 100. `window_mae` is also saved and must not be
substituted for the published aggregate MAE.

The relevant paper Table 1 comparison is **out-of-sample AQI-36 GRIN**:
MAE **12.08 ± 0.47**, MSE **523.14 ± 57.17**, MRE **17.00 ± 0.67%**. BRITS has
MAE 14.50 ± 0.35. The in-sample GRIN MAE 10.51 is a different experiment.
Local training-station-mean and linear-interpolation controls are also computed;
they are explicitly local controls, not claims to reproduce the paper's baseline
implementations. Linear interpolation uses visible past and future readings,
consistent with an offline imputation task.

## Runtime changes and saved evidence

The original stack used Python 3.8, PyTorch 1.8 and Lightning 1.4. This project
uses a current, pinned PyTorch environment and an independent training loop to
avoid obsolete Lightning/TensorFlow dependencies. Preprocessing replaces removed
NumPy `in1d` and deprecated pandas filling/grouping interfaces with their modern
equivalents. The downloaded GRIN model source is not modified or redistributed.
Different runtimes, one seed and especially the reduced training budget can
produce substantial differences from the paper; those differences must remain
visible in the report.

Each completed run writes its full configuration, source/data provenance,
per-epoch history, best checkpoint, aggregate test predictions, metrics and an
artifact-checksum manifest. Cached results are loaded only when configuration,
data fingerprint, upstream source hashes and saved-artifact hashes agree.
The executed notebook records the actual output and comparisons rather than
substituting the published score.

Public API: `prepare_aqi36(data_dir)` returns `PreparedData`;
`run_experiment(data, Config(), output_dir)` trains and evaluates;
`load_run(data, config, output_dir)` verifies and reads a completed run. The
notebook provides the ordered executable workflow and the README provides the
environment and launch commands.
