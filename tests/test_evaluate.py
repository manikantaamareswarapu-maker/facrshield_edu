from pipelines.shared.evaluate import (
    compute_auroc, compute_auprc,
    compute_f1_at_best_threshold, compute_ece
)
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_auroc_perfect():
    scores = [0.9, 0.8, 0.2, 0.1]
    labels = [1, 1, 0, 0]
    assert compute_auroc(scores, labels) == 1.0


def test_auroc_random():
    scores = [0.5, 0.5, 0.5, 0.5]
    labels = [1, 0, 1, 0]
    auroc = compute_auroc(scores, labels)
    assert 0.0 <= auroc <= 1.0


def test_auroc_in_range():
    np.random.seed(42)
    scores = np.random.rand(100).tolist()
    labels = np.random.randint(0, 2, 100).tolist()
    auroc = compute_auroc(scores, labels)
    assert 0.0 <= auroc <= 1.0


def test_auprc_in_range():
    np.random.seed(42)
    scores = np.random.rand(100).tolist()
    labels = np.random.randint(0, 2, 100).tolist()
    auprc = compute_auprc(scores, labels)
    assert 0.0 <= auprc <= 1.0


def test_f1_threshold_on_val_not_test():
    val_scores = [0.9, 0.8, 0.2, 0.1]
    val_labels = [1, 1, 0, 0]
    test_scores = [0.7, 0.6, 0.3, 0.2]
    test_labels = [1, 1, 0, 0]
    f1, thresh = compute_f1_at_best_threshold(
        val_scores, val_labels, test_scores, test_labels
    )
    assert 0.0 <= f1 <= 1.0
    assert 0.0 <= thresh <= 1.0


def test_f1_perfect():
    val_scores = [0.9, 0.1]
    val_labels = [1, 0]
    test_scores = [0.9, 0.1]
    test_labels = [1, 0]
    f1, _ = compute_f1_at_best_threshold(
        val_scores, val_labels, test_scores, test_labels
    )
    assert f1 == 1.0


def test_ece_in_range():
    probs = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([1, 1, 0, 0])
    ece = compute_ece(probs, labels)
    assert 0.0 <= ece <= 1.0
