import numpy as np
from pipelines.token_prob.scorer import (
    compute_perplexity, compute_entropy, extract_features
)
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_perplexity_positive():
    log_probs = np.array([-1.0, -2.0, -0.5, -1.5])
    ppl = compute_perplexity(log_probs)
    assert ppl > 0


def test_perplexity_decreases_with_confidence():
    high_conf = np.array([-0.1, -0.1, -0.1])
    low_conf = np.array([-3.0, -3.0, -3.0])
    assert compute_perplexity(high_conf) < compute_perplexity(low_conf)


def test_entropy_returns_two_values():
    log_probs = np.array([-1.0, -2.0, -0.5])
    mean_ent, max_ent = compute_entropy(log_probs)
    assert isinstance(mean_ent, float)
    assert isinstance(max_ent, float)


def test_entropy_max_geq_mean():
    log_probs = np.array([-1.0, -2.0, -0.5, -3.0])
    mean_ent, max_ent = compute_entropy(log_probs)
    assert max_ent >= mean_ent


def test_extract_features_keys():
    row = {
        'token_logprobs': np.array([-1.0, -2.0, -0.5, -1.5, -0.8,
                                    -1.2, -0.9, -1.1, -2.0, -0.7],
                                   dtype=np.float32).tobytes(),
        'seq_len': 10
    }
    feats = extract_features(row)
    for key in ['perplexity', 'mean_entropy', 'max_entropy', 'tail_mean_nll', 'sent_length']:
        assert key in feats


def test_extract_features_values_finite():
    row = {
        'token_logprobs': np.array([-1.0, -2.0, -0.5, -1.5, -0.8,
                                    -1.2, -0.9, -1.1, -2.0, -0.7],
                                   dtype=np.float32).tobytes(),
        'seq_len': 10
    }
    feats = extract_features(row)
    for key, val in feats.items():
        if isinstance(val, float):
            assert np.isfinite(val), f"{key} is not finite"
