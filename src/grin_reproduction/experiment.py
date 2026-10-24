"""Reproducible CPU/GPU training of the pinned authors' GRIN architecture."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
import platform
from pathlib import Path
import random
import time

import numpy as np
import torch

from .upstream import COMMIT, MANIFEST, make_model


@dataclass
class Config:
    # bounded changes only the training budget, keeping the published architecture.
    profile: str = "bounded"
    seed: int = 42
    max_epochs: int = 30
    samples_per_epoch: int = 512
    batch_size: int = 32
    patience: int = 40
    learning_rate: float = .001
    min_learning_rate: float = .0001
    whiten_probability: float = .05
    gradient_clip: float = 5.
    hidden_size: int = 64
    embedding_size: int = 8
    num_threads: int = 1
    device: str = "cpu"

    @classmethod
    def paper(cls, seed=42, device="cpu"):
        return cls(profile="paper", seed=seed, max_epochs=300,
                   samples_per_epoch=5120, device=device)


def seed_everything(seed):
    # Required by deterministic CUDA matrix products; has no effect on CPU runs.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def data_fingerprint(data):
    """Bind cached results to exact values, masks, graph, split and normalization."""
    digest = hashlib.sha256()
    for name in ("values", "input_mask", "eval_mask", "adjacency", "train_starts", "val_starts", "test_starts"):
        array = np.ascontiguousarray(getattr(data, name))
        digest.update(name.encode())
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    digest.update(str((float(data.mean), float(data.std), data.window)).encode())
    return digest.hexdigest()


def load_run(data, config, output_dir):
    """Load saved results only when configuration, source, data and file hashes match."""
    output = Path(output_dir)
    recorded = json.loads((output / "config.json").read_text())
    provenance = json.loads((output / "provenance.json").read_text())
    if recorded != asdict(config) or provenance["data_fingerprint"] != data_fingerprint(data):
        raise ValueError("Cached run configuration/data differ; choose a fresh output directory")
    if provenance["upstream_commit"] != COMMIT or provenance["source_sha256"] != MANIFEST:
        raise ValueError("Cached upstream model version differs")
    integrity = json.loads((output / "artifact_hashes.json").read_text())
    for name, expected in integrity.items():
        if hashlib.sha256((output / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Cached run artifact checksum mismatch: {name}")
    return json.loads((output / "metrics.json").read_text())


def masked_mae(prediction, target, mask):
    """Exclude absent targets rather than assigning zero-valued target measurements."""
    if not bool(mask.any()):
        raise ValueError("Cannot score an empty target mask")
    return (prediction - target).abs().masked_select(mask).mean()


def five_term_loss(imputation, predictions, target, original_mask):
    """Equation 1 training objective: final + four directional/stage readouts."""
    terms = [masked_mae(imputation, target, original_mask)]
    terms += [masked_mae(prediction, target, original_mask) for prediction in predictions]
    if len(terms) != 5:
        raise ValueError("GRIN must supply one final and four intermediate predictions")
    return torch.stack(terms).sum()


def window_batch(data, starts, device="cpu"):
    indices = np.asarray(starts)[:, None] + np.arange(data.window)[None, :]
    values = torch.as_tensor(data.values[indices, :, None], device=device)
    mask = torch.as_tensor(data.input_mask[indices, :, None], device=device)
    target_mask = torch.as_tensor(data.eval_mask[indices, :, None], device=device)
    mean = torch.as_tensor(data.mean, dtype=torch.float32, device=device)
    std = torch.as_tensor(data.std, dtype=torch.float32, device=device)
    normalized = (values - mean) / std
    # Values at missing positions never enter the model, including hidden test truths.
    x = torch.where(mask, normalized, 0.)
    return x, mask, normalized, target_mask, indices


@torch.no_grad()
def evaluate(model, data, starts, batch_size=32, device="cpu", return_arrays=False):
    """Report both Lightning-style window MAE and authors' final timestamp-mean MAE."""
    model.eval()
    sums = np.zeros(data.values.shape, dtype=np.float64)
    counts = np.zeros(len(data.values), dtype=np.int64)
    absolute_sum, target_count = 0., 0
    mean, std = np.asarray(data.mean), np.asarray(data.std)
    for offset in range(0, len(starts), batch_size):
        x, mask, _, targets, indices = window_batch(data, starts[offset:offset + batch_size], device)
        prediction = model(x, mask=mask).cpu().numpy()[..., 0] * std + mean
        truth = data.values[indices]
        eval_mask = targets.cpu().numpy()[..., 0]
        absolute_sum += np.abs(prediction - truth)[eval_mask].sum(dtype=np.float64)
        target_count += int(eval_mask.sum())
        np.add.at(sums, indices.reshape(-1), prediction.reshape(-1, prediction.shape[-1]))
        np.add.at(counts, indices.reshape(-1), 1)
    covered = counts > 0
    aggregated = np.divide(sums, counts[:, None], out=np.full_like(sums, np.nan), where=counts[:, None] > 0)
    score_mask = data.eval_mask & covered[:, None]
    if not score_mask.any() or not target_count:
        raise ValueError("Evaluation split has no held-out targets")
    errors = aggregated[score_mask] - data.values[score_mask]
    result = dict(mae=float(np.abs(errors).mean()), mse=float(np.square(errors).mean()),
                  rmse=float(np.sqrt(np.square(errors).mean())),
                  mre=float(np.abs(errors).sum() / np.abs(data.values[score_mask]).sum()),
                  window_mae=float(absolute_sum / target_count),
                  n_targets=int(score_mask.sum()), n_windows=len(starts))
    if return_arrays:
        return result, aggregated.astype(np.float32), score_mask
    return result


def evaluate_baselines(data):
    """Simple controls use visible inputs and training-only statistics, never test truths."""
    import pandas as pd
    coverage = np.zeros(len(data.values), dtype=bool)
    for start in data.test_starts:
        coverage[start:start + data.window] = True
    test_mask = data.eval_mask & coverage[:, None]
    train_coverage = np.zeros(len(data.values), dtype=bool)
    for start in data.train_starts:
        train_coverage[start:start + data.window] = True
    visible_train = data.input_mask & train_coverage[:, None]
    sums = np.where(visible_train, data.values, 0.).sum(0)
    denominators = visible_train.sum(0)
    global_mean = float(np.where(visible_train, data.values, 0.).sum() / visible_train.sum())
    station_mean = np.divide(sums, denominators, out=np.full_like(sums, global_mean), where=denominators > 0)
    mean_prediction = np.broadcast_to(station_mean, data.values.shape)
    visible = pd.DataFrame(np.where(data.input_mask, data.values, np.nan))
    # Bidirectional interpolation is appropriate to offline imputation, like GRIN.
    interpolation = visible.interpolate(axis=0, limit_direction="both").fillna(pd.Series(station_mean)).to_numpy()
    result = {}
    for name, prediction in (("Training station mean", mean_prediction), ("Linear interpolation", interpolation)):
        errors = prediction[test_mask] - data.values[test_mask]
        result[name] = dict(mae=float(np.abs(errors).mean()), mse=float(np.square(errors).mean()),
                            n_targets=int(test_mask.sum()))
    return result


def benchmark_batch(data, config=None):
    """Measure one actual forward/backward update before committing to a run budget."""
    config = config or Config()
    torch.set_num_threads(config.num_threads)
    seed_everything(config.seed)
    model = make_model(data.adjacency, config.hidden_size, config.embedding_size).to(config.device)
    model.train()
    starts = data.train_starts[:config.batch_size]
    x, mask, target, _, _ = window_batch(data, starts, config.device)
    start = time.perf_counter()
    imputation, predictions = model(x, mask=mask)
    loss = five_term_loss(imputation, predictions, target, mask)
    loss.backward()
    seconds = time.perf_counter() - start
    return dict(seconds_per_batch=seconds, parameter_count=sum(p.numel() for p in model.parameters()),
                loss=float(loss.detach()), batch_size=len(starts))


def run_experiment(data, config, output_dir):
    """Train, select on validation, and evaluate once on held-out AQI-36 test months."""
    if config.profile == "paper" and (config.max_epochs != 300 or config.samples_per_epoch != 5120):
        raise ValueError("The paper profile must retain its 300 epoch / 5120 sample budget")
    if config.max_epochs < 1 or config.samples_per_epoch < 1:
        raise ValueError("Training budget must be positive")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metrics.json").exists():
        raise FileExistsError(f"Completed result already exists: {output}; choose a new directory")
    seed_everything(config.seed)
    torch.set_num_threads(config.num_threads)
    model = make_model(data.adjacency, config.hidden_size, config.embedding_size).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_epochs, eta_min=config.min_learning_rate)
    rng = np.random.default_rng(config.seed)
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    provenance = dict(upstream_commit=COMMIT, source_sha256=MANIFEST, data_fingerprint=data_fingerprint(data),
                      python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                      platform=platform.platform(), data=getattr(data, "provenance", {}),
                      loss="sum of five masked MAEs on original visible training targets, scaled units",
                      selection="minimum validation window-weighted MAE, original units",
                      final_scoring="overlapping test predictions averaged per timestamp, then masked MAE",
                      deviations=(["One seed rather than five published runs"] if config.profile == "paper" else ["Reduced training budget; not a full paper reproduction", "Cosine schedule compressed to the selected epoch budget", "One seed rather than five published runs"]))
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, default=str) + "\n")
    best, best_epoch, stale = float("inf"), 0, 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, config.max_epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        sampled = rng.choice(data.train_starts, size=config.samples_per_epoch, replace=True)
        train_sum, batches = 0., 0
        for offset in range(0, len(sampled), config.batch_size):
            x, original_mask, target, _, _ = window_batch(data, sampled[offset:offset + config.batch_size], config.device)
            mask = original_mask & (torch.rand_like(x) >= config.whiten_probability)
            x = torch.where(mask, x, 0.)
            optimizer.zero_grad(set_to_none=True)
            imputation, predictions = model(x, mask=mask)
            loss = five_term_loss(imputation, predictions, target, original_mask)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            train_sum += float(loss.detach())
            batches += 1
        validation = evaluate(model, data, data.val_starts, config.batch_size, config.device)
        row = dict(epoch=epoch, train_five_term_mae=train_sum / batches,
                   val_window_mae=validation["window_mae"], val_aggregated_mae=validation["mae"],
                   learning_rate=optimizer.param_groups[0]["lr"], seconds=time.perf_counter() - epoch_start)
        history.append(row)
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(f"Epoch {epoch:3d}: loss={row['train_five_term_mae']:.4f}, val MAE={row['val_window_mae']:.3f}, {row['seconds']:.1f}s", flush=True)
        if validation["window_mae"] < best:
            best, best_epoch, stale = validation["window_mae"], epoch, 0
            torch.save(dict(state_dict=model.state_dict(), config=asdict(config), epoch=epoch,
                            upstream_commit=COMMIT, mean=np.asarray(data.mean).tolist(), std=np.asarray(data.std).tolist()), output / "best_model.pt")
        else:
            stale += 1
        scheduler.step()
        if stale >= config.patience:
            break
    checkpoint = torch.load(output / "best_model.pt", map_location=config.device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    metrics, predictions, mask = evaluate(model, data, data.test_starts, config.batch_size, config.device, True)
    metrics.update(best_epoch=best_epoch, best_validation_window_mae=best,
                   elapsed_seconds=time.perf_counter() - started, epochs_completed=len(history),
                   optimizer_updates=sum(int(np.ceil(config.samples_per_epoch / config.batch_size)) for _ in history),
                   parameter_count=sum(p.numel() for p in model.parameters()), seed=config.seed,
                   profile=config.profile, published_grin_mae=12.08, published_grin_mae_std=.47,
                   difference_from_published_mae=metrics["mae"] - 12.08,
                   baselines=evaluate_baselines(data))
    np.savez_compressed(output / "test_predictions.npz", prediction=predictions, evaluation_mask=mask,
                        timestamps=np.asarray(data.timestamps).astype(str))
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    files = ("config.json", "provenance.json", "history.json", "metrics.json", "best_model.pt", "test_predictions.npz")
    (output / "artifact_hashes.json").write_text(json.dumps({name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in files}, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics
