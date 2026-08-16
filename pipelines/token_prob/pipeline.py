import time
from datasets import load_dataset
from pipelines.token_prob.scorer import extract_features_vectorized


def run_pipeline(model_name='bart', dataset_name='cnndm'):
    scores_ds = load_dataset(
        'factshield-team/cache',
        f'{model_name}_{dataset_name}_token_scores'
    )['train']
    scores_df = scores_ds.to_pandas()

    labels_ds = load_dataset('factshield-team/faithbench-labels')['train']
    labels_df = labels_ds.to_pandas()

    start = time.time()

    features_df = extract_features_vectorized(scores_df)
    features_df['summarizer'] = model_name
    features_df['dataset'] = dataset_name

    elapsed = time.time() - start
    print(f"✓ {len(features_df)} docs processed in {elapsed:.1f}s")

    merged = features_df.merge(
        labels_df[['doc_id', 'label_binary_worst', 'label_binary_best']],
        on='doc_id',
        how='left'
    )

    return merged, elapsed
