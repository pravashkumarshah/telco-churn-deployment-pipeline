"""FastAPI application for secure telecom churn predictions."""
from __future__ import annotations
import json, logging, os, secrets, time, uuid
from functools import lru_cache
from pathlib import Path
import joblib, pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.security import APIKeyHeader
from app.config import MODEL_PATH, PSI_ALERT_THRESHOLD, PROJECT_ROOT
from app.drift import calculate_psi
from app.schemas import BatchPredictionRequest, BatchPredictionResponse, CustomerFeatures, DriftCheckRequest, PredictionResponse
logger=logging.getLogger("churn_api"); logger.setLevel(logging.INFO)
if not logger.handlers:
    handler=logging.StreamHandler(); handler.setFormatter(logging.Formatter("%(message)s")); logger.addHandler(handler)
api_key_header=APIKeyHeader(name="X-API-Key",auto_error=False)
app=FastAPI(title="Telecom Churn Prediction API",version="1.0.0",description="Secure churn scoring service with versioned sklearn pipeline and validation-selected business-cost threshold.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

INDEX_HTML_PATH = PROJECT_ROOT / "app" / "static" / "index.html"

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def serve_frontend():
    if INDEX_HTML_PATH.is_file():
        return HTMLResponse(content=INDEX_HTML_PATH.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Telecom Churn API</h1><p>Frontend template not found.</p>")

@lru_cache(maxsize=1)
def get_bundle() -> dict:
    path=Path(os.getenv("MODEL_PATH",str(MODEL_PATH)))
    if not path.is_file(): raise FileNotFoundError(f"Model artefact not found at {path}. Run scripts/train.py first.")
    bundle=joblib.load(path); required={"model","threshold","model_version","metadata"}; missing=required-set(bundle)
    if missing: raise RuntimeError(f"Model artefact is missing keys: {sorted(missing)}")
    return bundle

def _ready_bundle() -> dict:
    try: return get_bundle()
    except (FileNotFoundError,RuntimeError) as exc: raise HTTPException(status_code=503,detail=str(exc)) from exc

def require_api_key(api_key: str|None=Security(api_key_header)) -> str:
    expected=os.getenv("CHURN_API_KEY")
    if not expected: raise HTTPException(status_code=503,detail="API key authentication is not configured")
    if not api_key or not secrets.compare_digest(api_key,expected): raise HTTPException(status_code=401,detail="Missing or invalid API key")
    return api_key

@app.middleware("http")
async def structured_request_logging(request:Request,call_next):
    request_id=request.headers.get("X-Request-ID",str(uuid.uuid4())); started=time.perf_counter(); status_code=500
    try:
        response=await call_next(request); status_code=response.status_code; response.headers["X-Request-ID"]=request_id; return response
    finally:
        logger.info(json.dumps({"event":"http_request","request_id":request_id,"method":request.method,"path":request.url.path,"status_code":status_code,"latency_ms":round((time.perf_counter()-started)*1000,3)},separators=(",",":")))

def _predict_one(customer:CustomerFeatures,bundle:dict)->PredictionResponse:
    frame=pd.DataFrame([customer.model_dump()]); probability=float(bundle["model"].predict_proba(frame)[0,1]); threshold=float(bundle["threshold"]); churn=probability>=threshold
    return PredictionResponse(churn_probability=round(probability,6),churn=churn,threshold=threshold,model_version=str(bundle["model_version"]),recommendation="prioritise_for_retention_review" if churn else "standard_customer_journey")

@app.get("/health",tags=["operations"])
def health():
    bundle=_ready_bundle(); return {"status":"ok","model_version":bundle["model_version"]}

@app.post("/predict",response_model=PredictionResponse,tags=["prediction"])
def predict(customer:CustomerFeatures,_:str=Depends(require_api_key)):
    bundle=_ready_bundle(); started=time.perf_counter(); result=_predict_one(customer,bundle)
    logger.info(json.dumps({"event":"prediction","model_version":bundle["model_version"],"latency_ms":round((time.perf_counter()-started)*1000,3)},separators=(",",":"))); return result

@app.post("/predict/batch",response_model=BatchPredictionResponse,tags=["prediction"])
def predict_batch(request:BatchPredictionRequest,_:str=Depends(require_api_key)):
    bundle=_ready_bundle(); started=time.perf_counter(); results=[_predict_one(customer,bundle) for customer in request.customers]
    logger.info(json.dumps({"event":"batch_prediction","batch_size":len(results),"model_version":bundle["model_version"],"latency_ms":round((time.perf_counter()-started)*1000,3)},separators=(",",":")))
    return BatchPredictionResponse(predictions=results,count=len(results),model_version=str(bundle["model_version"]))

@app.post("/drift/check",tags=["monitoring"])
def drift_check(request:DriftCheckRequest,_:str=Depends(require_api_key)):
    bundle=_ready_bundle(); profiles=bundle["metadata"].get("drift_reference")
    if not profiles: raise HTTPException(status_code=503,detail="Drift reference profile is unavailable")
    frame=pd.DataFrame([customer.model_dump() for customer in request.recent_customers]); results=calculate_psi(profiles,frame)
    flagged=[name for name,item in results.items() if item.get("psi") is not None and item["psi"]>PSI_ALERT_THRESHOLD]
    return {"model_version":bundle["model_version"],"psi_alert_threshold":PSI_ALERT_THRESHOLD,"flagged_features":flagged,"status":"drift_detected" if flagged else "no_material_drift_detected","features":results,"sample_size":len(frame),"note":"PSI is a screening signal; investigate source, seasonality and data quality before retraining."}
