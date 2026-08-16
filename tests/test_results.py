import os
from dotenv import load_dotenv
from datasets import load_dataset
from huggingface_hub import login

load_dotenv()
login(token=os.environ['HF_TOKEN'])

MODELS = ['bart', 't5', 'pegasus']
DATASETS = ['cnndm', 'xsum', 'faithbench']


def test_task1_exists():
    ds = load_dataset('factshield-team/results', 'task1_scores')['train']
    assert len(ds) > 0


def test_task1_columns():
    ds = load_dataset('factshield-team/results', 'task1_scores')['train']
    for col in ['doc_id', 'sent_id', 'summarizer', 'dataset', 'K', 'score_mode', 'score']:
        assert col in ds.column_names


def test_task1_all_combos():
    df = load_dataset('factshield-team/results', 'task1_scores')['train'].to_pandas()
    for model in MODELS:
        for dataset in DATASETS:
            subset = df[(df['summarizer'] == model) & (df['dataset'] == dataset)]
            assert len(subset) > 0, f"Missing task1: {model} x {dataset}"


def test_task1_score_modes():
    df = load_dataset('factshield-team/results', 'task1_scores')['train'].to_pandas()
    assert set(df['score_mode'].unique()) >= {'bert', 'nli', 'ngram'}


def test_task2_exists():
    ds = load_dataset('factshield-team/results', 'task2_scores')['train']
    assert len(ds) > 0


def test_task2_columns():
    ds = load_dataset('factshield-team/results', 'task2_scores')['train']
    for col in ['doc_id', 'summarizer', 'dataset', 'method', 'score']:
        assert col in ds.column_names


def test_task2_all_combos():
    df = load_dataset('factshield-team/results', 'task2_scores')['train'].to_pandas()
    for model in MODELS:
        for dataset in DATASETS:
            subset = df[(df['summarizer'] == model) & (df['dataset'] == dataset)]
            assert len(subset) > 0, f"Missing task2: {model} x {dataset}"


def test_task2_scores_in_range():
    df = load_dataset('factshield-team/results', 'task2_scores')['train'].to_pandas()
    assert df['score'].between(0.0, 1.0).all(), "MiniCheck scores should be between 0 and 1"


def test_task3_exists():
    ds = load_dataset('factshield-team/results', 'task3_scores')['train']
    assert len(ds) > 0


def test_task3_columns():
    ds = load_dataset('factshield-team/results', 'task3_scores')['train']
    for col in ['doc_id', 'summarizer', 'dataset', 'perplexity',
                'mean_entropy', 'max_entropy', 'tail_mean_nll', 'sent_length']:
        assert col in ds.column_names


def test_task3_all_combos():
    df = load_dataset('factshield-team/results', 'task3_scores')['train'].to_pandas()
    for model in MODELS:
        for dataset in DATASETS:
            subset = df[(df['summarizer'] == model) & (df['dataset'] == dataset)]
            assert len(subset) > 0, f"Missing task3: {model} x {dataset}"


def test_task3_perplexity_positive():
    df = load_dataset('factshield-team/results', 'task3_scores')['train'].to_pandas()
    assert (df['perplexity'] > 0).all(), "Perplexity should be positive"


def test_final_table_exists():
    ds = load_dataset('factshield-team/results', 'final_table')['train']
    assert len(ds) > 0


def test_final_table_columns():
    ds = load_dataset('factshield-team/results', 'final_table')['train']
    for col in ['method', 'summarizer', 'dataset', 'AUROC', 'AUPRC']:
        assert col in ds.column_names


def test_final_table_auroc_in_range():
    df = load_dataset('factshield-team/results', 'final_table')['train'].to_pandas()
    assert df['AUROC'].between(0.0, 1.0).all(), "AUROC should be between 0 and 1"


def test_final_table_has_all_methods():
    df = load_dataset('factshield-team/results', 'final_table')['train'].to_pandas()
    expected = {'selfcheck_bert', 'selfcheck_nli', 'minicheck', 'token_prob_classifier'}
    assert expected.issubset(set(df['method'].unique()))
