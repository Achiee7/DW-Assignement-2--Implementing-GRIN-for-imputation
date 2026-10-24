"""Meaningful leakage, model and scoring checks on small artificial fixtures.

These fixtures test code properties; they are never reported as experimental results.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from grin_reproduction.experiment import evaluate, five_term_loss, window_batch
from grin_reproduction.upstream import make_model


def fixture_data():
    values = np.arange(24, dtype=np.float32).reshape(8, 3)
    mask = np.ones_like(values, dtype=bool)
    mask[2:5, 1] = False
    return SimpleNamespace(values=values, input_mask=mask, eval_mask=~mask,
                           mean=0., std=1., window=3)


def test_window_batch_does_not_reveal_hidden_ground_truth():
    data = fixture_data()
    x1, mask, target1, _, _ = window_batch(data, [1, 2])
    data.values[~data.input_mask] += 10000
    x2, _, target2, _, _ = window_batch(data, [1, 2])
    assert torch.equal(x1, x2)
    assert not torch.equal(target1, target2)
    assert (x1[~mask] == 0).all()


def test_original_model_outputs_and_observed_value_preservation():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    adjacency = np.ones((3, 3), dtype=np.float32) - np.eye(3, dtype=np.float32)
    model = make_model(adjacency)
    x = torch.randn(2, 4, 3, 1)
    mask = torch.rand_like(x) > .3
    model.train()
    final, intermediate = model(x, mask=mask)
    assert final.shape == x.shape
    assert intermediate.shape == (4, *x.shape)
    loss = five_term_loss(final, intermediate, x, mask)
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval()
    with torch.no_grad():
        a = model(x, mask=mask)
        changed = torch.where(mask, x, torch.full_like(x, 123456.))
        b = model(changed, mask=mask)
    assert torch.equal(a[mask], x[mask])
    assert torch.equal(a, b)


def test_five_terms_mask_missing_targets():
    prediction = torch.zeros(1, 2, 1, 1)
    target = torch.tensor([1., 999.]).view_as(prediction)
    mask = torch.tensor([True, False]).view_as(prediction)
    assert five_term_loss(prediction, prediction.repeat(4, 1, 1, 1, 1), target, mask).item() == 5.
    with pytest.raises(ValueError, match="empty"):
        five_term_loss(prediction, prediction.repeat(4, 1, 1, 1, 1), target, mask & False)


def test_evaluation_scores_only_covered_hidden_targets_and_averages_overlap():
    data = fixture_data()
    class ZeroImputer(torch.nn.Module):
        def forward(self, x, mask):
            return torch.where(mask, x, 0.)
    results, predictions, scored = evaluate(ZeroImputer(), data, np.array([1, 2]), return_arrays=True)
    expected = data.eval_mask.copy()
    expected[5:] = False
    expected[:1] = False
    assert np.array_equal(scored, expected)
    assert results["n_targets"] == 3
    assert results["mae"] == float(data.values[scored].mean())
    assert np.isnan(predictions[0]).all()
