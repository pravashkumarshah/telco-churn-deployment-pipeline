# Telecom Churn Model Deployment Pipeline

A reproducible educational reference implementation that moves a telecom churn classifier from a notebook into a secured prediction API. It includes a leakage-safe scikit-learn preprocessing/model pipeline, cross-validated hyperparameter search, a validation-selected business-cost threshold, versioned model metadata, FastAPI/Pydantic endpoints, PSI drift screening, Docker packaging, and automated tests.

> **Status:** Fully verified reference implementation. Model training, validation-selected cost threshold (0.32), interactive Web UI console, 28/28 unit and integration tests, automated browser audit, and sub-100 ms live benchmarks (40.94 ms median latency) are verified and documented in `reports/`.

## Dataset
Uses the public IBM Telco Customer Churn sample (7,043 rows, 21 columns). `TotalCharges` is text in the source and is converted to numeric in the pipeline. `customerID` is checked for duplicates and excluded from features. Download URL: https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv . The download script verifies SHA-256 before saving. Check the dataset source terms before reuse.

## Architecture

```text
CSV -> SHA-256 checksum -> stratified split (train: 60%, val: 20%, test: 20%)
    -> sklearn Pipeline (cleaning, imputation, OneHotEncoding, StandardScaler, RandomForest)
    -> RandomizedSearchCV / StratifiedKFold -> validation-only cost cutoff (0.32)
    -> test holdout metrics (ROC-AUC: 0.8426, Churn Recall: 89.04%) + PSI drift baseline
FastAPI Backend -> API Key Authentication + Pydantic validation
                -> Single & Batch Predictions + PSI drift screening
                -> Modern Web UI Console (Meridian Telecom Retention Suite)
```

## Quick Start (Local Setup)

Requires Python 3.12 or 3.13.

### 1. Environment & Training
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

# Install dependencies
python -m pip install --upgrade pip
pip install -r requirements-dev.txt

# Download dataset (SHA-256 verified) and train model
python scripts/download_data.py
python scripts/train.py --n-iter 12 --cv-folds 3

# Run tests
pytest -v
```

### 2. Run the Service
Create a `.env` file (or set environment variable):
```bash
# Windows (PowerShell):
$env:CHURN_API_KEY = "your-secure-secret-key"
uvicorn app.main:app --host 127.0.0.1 --port 8000 --env-file .env

# macOS / Linux:
export CHURN_API_KEY="your-secure-secret-key"
uvicorn app.main:app --host 127.0.0.1 --port 8000 --env-file .env
```

Open your browser:
- **Interactive Web UI**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Swagger API Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health Telemetry**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

## Docker & Container Deployment

### Option A: Using Docker Compose (Recommended)
```bash
# Start container with volume-mounted model artifacts
docker compose up -d

# Check logs
docker compose logs -f

# Verify container health and endpoints
python scripts/verify_container.py

# Stop container
docker compose down
```

### Option B: Using Docker CLI
```bash
# 1. Build the lightweight multi-stage image
docker build -t telco-churn-api:1.0 .

# 2. Verify image size (< 600 MB)
./scripts/check_container_size.sh telco-churn-api:1.0 600

# 3. Run container as non-root user with read-only model volume
docker run -d \
  --name telco-churn-api \
  -p 8000:8000 \
  -e CHURN_API_KEY="your-secure-secret-key" \
  -v "$(pwd)/artifacts:/app/artifacts:ro" \
  telco-churn-api:1.0

# 4. Run automated verification suite
python scripts/verify_container.py

# 5. Graceful shutdown
docker stop -t 10 telco-churn-api
```

> **Why mount `/app/artifacts:ro`?**  
> Mounting artifacts at runtime keeps the Docker image under 600 MB, enables zero-downtime model hot-swapping without rebuilding the container, and mounts model files in read-only mode for security.

## API Endpoints

All prediction and drift endpoints require the `X-API-Key` header; `/health` and `/` (UI) are unauthenticated.

| Method | Path | Auth Required | Purpose |
|---|---|---|---|
| GET | `/` | No | Modern Web UI Console |
| GET | `/health` | No | Service readiness & active model version |
| POST | `/predict` | Yes | Individual customer prediction & recommendation |
| POST | `/predict/batch` | Yes | Batch prediction (1–100 customer records) |
| POST | `/drift/check` | Yes | Population Stability Index (PSI) drift diagnostics |
| GET | `/docs` | No | Interactive OpenAPI / Swagger UI |
| GET | `/openapi.json` | No | Raw OpenAPI schema |

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
