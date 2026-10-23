<div align="center">

# AI Training · Prediction API

**AutoGluon regression experiments with a FastAPI service and CSV prediction tool.**

[API reference](#prediction-api) · [Quick start](#quick-start) · [Tests](#tests) · [Remaining work](#remaining-work)

</div>

> Experimental project. API and CSV contracts are tested with synthetic predictors.
> Model accuracy, dataset quality and production deployment have not been validated by these tests.

## What works

| Workflow | Behavior |
| --- | --- |
| Single prediction | A feature object returns one numeric `prediction` |
| Batch prediction | Both batch forms return `predictions` and `count`, even for one row |
| Readiness | `/health` returns 503 until a saved model has loaded; `/docs` remains available |
| Model location | `models/model` by default, resolved from the repository root |
| CSV prediction | Explicit input/output arguments; atomic output; source aliases rejected |
| Contract tests | No AutoGluon, TensorFlow, saved models or repository datasets required |

## Quick start

```bash
git clone https://github.com/art9762/prod.git
cd prod
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python model/predict.py serve --model-path models/model --port 8000
```

Supply a trusted AutoGluon predictor under `models/model`, or select its directory
with `--model-path` or the `MODEL_PATH` environment variable. Relative model paths
are resolved from the repository root, including when launched from another
working directory. Importing `model.predict` or requesting `--help` does not load
AutoGluon or read a model.

The service exposes [interactive API documentation](http://localhost:8000/docs).
If the model or inference runtime is unavailable, the service starts with
prediction/readiness endpoints returning 503. Fix the installation/path and
restart it. The API does not train or download a model automatically.

For the API shell and tests without the ML runtime, install
`requirements-api.txt` or `requirements-dev.txt` instead of the full ML requirements.

## Prediction API

The feature names and types must match the loaded model. `total_area` below is
an example, not a declaration of the trained feature schema.

**Single record — `POST /predict`:**

```json
{"total_area": 60.0}
```

```json
{"prediction": 12345.67}
```

**Batch — `POST /predict`:**

```json
{"records": [{"total_area": 60.0}, {"total_area": 80.0}]}
```

**Alternatively — `POST /predict/batch`:**

```json
[{"total_area": 60.0}, {"total_area": 80.0}]
```

Both batch forms return the same envelope:

```json
{"predictions": [12345.67, 23456.78], "count": 2}
```

Requests accept 1–1000 non-empty feature objects. Batch output preserves input
order and includes `count` for a single-row batch. Invalid inputs return 400/422,
an unavailable predictor returns 503, and malformed/non-finite model output
returns an explicit failure. No partial batch result is returned.

Inference runs in a worker thread and calls to the shared predictor are
serialized. This does not establish throughput or a production capacity limit.

## CSV prediction

```bash
python model/predict.py predict --model-path models/model \
  --input test_data.csv --output predictions.csv
python model/predict.py --help
```

The original no-argument command still selects `test_data.csv` and
`predictions.csv`. A successful command replaces output atomically; failed
inference preserves existing output. Output cannot be the input file, a symlink
to it or a hard link to it.

## Training and evaluation

`model/train.py` and `model/evaluate.py` are the existing experiment scripts.
Configure an appropriate dataset, target and model output before running them.
The training script currently contains a developer-specific data path and runs
at module scope; it is not a reusable library entry point. The current training
configuration writes `models/model` and uses `monthly_price_rub` as its target.
A loaded model's own feature/label metadata remains authoritative.

No repository dataset or saved model is used by the API contract test suite.

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q tests
python -m compileall -q model/predict.py tests
```

The GitHub workflow checks out only prediction source, tests and dependency
lists. Tests cover readiness failures, both batch forms, invalid requests/model
outputs, model path resolution and atomic CSV output without altering inputs.

## Remaining work

- Replace experiment-specific training/evaluation paths with explicit CLI configuration.
- Validate a real trained predictor and its feature/target contract on a held-out dataset.
- Choose reproducible AutoGluon/ML dependency versions for the deployment platform.
- Implement and verify a container deployment; the existing `DOCKERFILE` is empty.
- Add deployment authentication, operational limits and inference performance measurements.

No project license has been selected in this repository.
