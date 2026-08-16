# FactShield

Factual Consistency Analysis Toolkit for Hallucination Detection in LLM Summaries.

Overview
--------
FactShield provides tools and pipelines to evaluate factual consistency of model-generated summaries. It includes data loaders, evaluation pipelines (entailment, self-check, token-prob), and utilities to build labeled datasets.

Quick Start
-----------
1. Install dependencies:

	```bash
	pip install -r requirements.txt
	```

2. Prepare data under the `data/` directory (see `data/README` or `pipelines/shared/splitter.py`).

3. Run a pipeline (example — token-prob):

	```bash
	python -m pipelines.token_prob.run
	```

Running tests
-------------
Run the test suite with:

```bash
pytest -q
```

Repository layout
-----------------
- `pipelines/` — evaluation pipelines for entailment, self-check, and token-prob.
- `data/` — raw datasets and prepared splits.
- `configs/` — pipeline configuration files.
- `scripts/` — helper scripts for assembling results and running sweeps.
- `tests/` — unit tests.

Contributing
------------
Issues and pull requests are welcome. For major changes, open an issue to discuss first.

License
-------
See the `LICENSE` file for license details (if present).

Contact
-------
For questions, open an issue or contact the maintainer.
