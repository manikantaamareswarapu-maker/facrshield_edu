import math
import numpy as np
import pandas as pd


def compute_perplexity(log_probs: np.ndarray) -> float:
    return math.exp(-float(np.mean(log_probs)))


def compute_entropy(log_probs: np.ndarray) -> tuple[float, float]:
    probs = np.exp(log_probs)
    entropy_per_token = -(probs * log_probs)
    return float(np.mean(entropy_per_token)), float(np.max(entropy_per_token))


def extract_features(row) -> dict:
    log_probs = np.frombuffer(row['token_logprobs'], dtype=np.float32)
    perplexity = compute_perplexity(log_probs)
    mean_ent, max_ent = compute_entropy(log_probs)
    tail_mean_nll = float(-np.mean(log_probs[-5:]))
    return {
        'perplexity': perplexity,
        'mean_entropy': mean_ent,
        'max_entropy': max_ent,
        'tail_mean_nll': tail_mean_nll,
        'sent_length': int(row['seq_len']),
    }


def extract_features_vectorized(df) -> "pd.DataFrame":

    log_probs_list = [
        np.frombuffer(b, dtype=np.float32)
        for b in df['token_logprobs']
    ]

    perplexity = np.array([math.exp(-float(np.mean(lp))) for lp in log_probs_list])
    probs_list = [np.exp(lp) for lp in log_probs_list]
    entropy_list = [-(p * lp) for p, lp in zip(probs_list, log_probs_list)]
    mean_entropy = np.array([float(np.mean(e)) for e in entropy_list])
    max_entropy = np.array([float(np.max(e)) for e in entropy_list])
    tail_mean_nll = np.array([float(-np.mean(lp[-5:])) for lp in log_probs_list])

    out = df[['doc_id', 'seq_len']].copy()
    out['perplexity'] = perplexity
    out['mean_entropy'] = mean_entropy
    out['max_entropy'] = max_entropy
    out['tail_mean_nll'] = tail_mean_nll
    out['sent_length'] = df['seq_len'].values
    return out
