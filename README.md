# Telecom Churn Model Deployment Pipeline

A reproducible educational reference implementation that moves a telecom churn classifier from a notebook into a secured prediction API. It includes a leakage-safe scikit-learn preprocessing/model pipeline, cross-validated hyperparameter search, a validation-selected business-cost threshold, versioned model metadata, FastAPI/Pydantic endpoints, PSI drift screening, Docker packaging, and automated tests.

> **Status:** source code and synthetic-data tests are included. Real-data training and live latency measurements have not yet been run in this environment; no real-data scores or <100 ms latency claim is made.

## Dataset
Uses the public IBM Telco Customer Churn sample (7,043 rows, 21 columns). `TotalCharges` is text in the source and is converted to numeric in the pipeline. `customerID` is checked for duplicates and excluded from features. Download URL: https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv . The download script verifies SHA-256 before saving. Check the dataset source terms before reuse.

## Architecture

```text
CSV -> schema checks/hash -> stratified train/validation/test
   -> sklearn Pipeline (cleaning, imputation, encoding, scaling, RandomForest)
   -> RandomizedSearchCV/StratifiedKFold -> validation cost threshold
   -> holdout metrics + versioned joblib bundle + PSI baseline
CRM -> FastAPI -> API key + Pydantic -> single/batch predictions
                              -> structured latency logs / drift check
```

## Quick start
Python 3.12 or 3.13 recommended.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
python scripts/download_data.py
python scripts/train.py --n-iter 12 --cv-folds 3
pytest -q
```

Set `CHURN_API_KEY` to a long random secret, then run:

```bash
# PowerShell: $env:CHURN_API_KEY = "your-long-secret"
# macOS/Linux: export CHURN_API_KEY="your-long-secret"
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Training writes `artifacts/model_bundle.joblib`, `metadata.json`, `test_metrics.json`, and `experiment_summary.json`. Raw data and model artefacts are git-ignored.

## API
All prediction/drift endpoints require `X-API-Key`; `/health` is unauthenticated.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Readiness and model version |
| POST | `/predict` | One customer prediction |
| POST | `/predict/batch` | 1–100 customer predictions |
| POST | `/drift/check` | PSI screening for 20–1,000 recent rows |
| GET | `/docs` | Swagger UI |
| GET | `/openapi.json` | OpenAPI document |

Examples: `curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -H "X-API-Key: $CHURN_API_KEY" --data @examples/single_prediction.json`.

Pydantic returns clear 422 errors for invalid values. Unknown category strings are supported through `OneHotEncoder(handle_unknown="ignore")`. Request logs omit payloads, customer IDs, and API secrets.

## Threshold and drift
Illustrative costs: false negative = 250 units; false positive = 40 units. Threshold is selected on validation only by minimizing `FN*250 + FP*40`. Replace assumptions with approved business costs. PSI > 0.20 triggers investigation, not automatic retraining. PSI signals input shift and does not prove accuracy degradation.

## Docker

```bash
docker build -t telco-churn-api:1.0 .
docker run --rm -p 8000:8000 -e CHURN_API_KEY="your-secret" -v "$(pwd)/artifacts:/app/artifacts:ro" telco-churn-api:1.0
./scripts/check_container_size.sh telco-churn-api:1.0 600
```

The image is multi-stage and runs as a non-root user. The 600 MB target must be verified by building and inspecting the image locally.

## Performance and testing
Run tests using `pytest -q`. The included benchmark script warms up the service and measures median/p95 latency: `python scripts/benchmark_api.py --api-key "$CHURN_API_KEY" --iterations 1000`. The <100 ms target is not claimed until measured on target hardware. CI runs the test suite on pushes and pull requests.

## Safe retraining
1. Validate schema, label quality, source hash and time window.
2. Train a candidate separately; do not overwrite the current production bundle.
3. Compare time-based holdout metrics, calibration, segment errors and business cost.
4. Review drift and business-approved costs; require human sign-off.
5. Deploy candidate to staging/canary, monitor, then promote; retain previous artefact for rollback.

See [model_card.md](model_card.md) and [docs/THRESHOLD_AND_DRIFT.md](docs/THRESHOLD_AND_DRIFT.md).
