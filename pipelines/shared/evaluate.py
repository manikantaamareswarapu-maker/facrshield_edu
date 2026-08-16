from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from netcal.metrics import ECE


def compute_auroc(scores, labels):
    return roc_auc_score(labels, scores)


def compute_auprc(scores, labels):
    return average_precision_score(labels, scores)


def compute_f1_at_best_threshold(val_scores, val_labels,
                                 test_scores, test_labels):
    thresholds = sorted(set(val_scores))
    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        preds = [1 if s >= t else 0 for s in val_scores]
        f = f1_score(val_labels, preds, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    test_preds = [1 if s >= best_t else 0 for s in test_scores]
    return f1_score(test_labels, test_preds, zero_division=0), best_t


def compute_ece(probs, labels, n_bins=10):
    ece = ECE(n_bins)
    return ece.measure(probs, labels)


def compute_latency(start_time, end_time, n_queries):
    return (end_time - start_time) * 1000 / n_queries
