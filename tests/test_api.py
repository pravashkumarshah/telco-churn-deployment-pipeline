import numpy as np
from fastapi.testclient import TestClient
from app.drift import build_reference_profiles
from app.main import app
class FakeModel:
    def predict_proba(self,frame):
        p=np.full(len(frame),.72); return np.column_stack([1-p,p])
def _client(monkeypatch,bundle):
    monkeypatch.setenv("CHURN_API_KEY","test-secret-key"); monkeypatch.setattr("app.main.get_bundle",lambda:bundle); return TestClient(app)
def test_health_endpoint_returns_model_version(monkeypatch):
    c=_client(monkeypatch,{"model":FakeModel(),"threshold":.4,"model_version":"v-test","metadata":{}}); r=c.get("/health"); assert r.status_code==200 and r.json()["model_version"]=="v-test"
def test_single_prediction_requires_api_key(monkeypatch,sample_customer):
    c=_client(monkeypatch,{"model":FakeModel(),"threshold":.4,"model_version":"v-test","metadata":{}}); assert c.post("/predict",json=sample_customer).status_code==401
def test_single_prediction_rejects_wrong_api_key(monkeypatch,sample_customer):
    c=_client(monkeypatch,{"model":FakeModel(),"threshold":.4,"model_version":"v-test","metadata":{}}); assert c.post("/predict",json=sample_customer,headers={"X-API-Key":"wrong"}).status_code==401
def test_single_prediction_returns_threshold_decision(monkeypatch,sample_customer):
    c=_client(monkeypatch,{"model":FakeModel(),"threshold":.4,"model_version":"v-test","metadata":{}}); r=c.post("/predict",json=sample_customer,headers={"X-API-Key":"test-secret-key"}); assert r.status_code==200 and r.json()["churn"] is True and r.json()["churn_probability"]==.72
def test_single_prediction_accepts_unseen_category(monkeypatch,sample_customer):
    c=_client(monkeypatch,{"model":FakeModel(),"threshold":.8,"model_version":"v-test","metadata":{}}); payload=dict(sample_customer,PaymentMethod="new CRM payment option"); r=c.post("/predict",json=payload,headers={"X-API-Key":"test-secret-key"}); assert r.status_code==200 and r.json()["churn"] is False
def test_invalid_monthly_charges_returns_clear_422(monkeypatch,sample_customer):
    c=_client(monkeypatch,{"model":FakeModel(),"threshold":.4,"model_version":"v-test","metadata":{}}); payload=dict(sample_customer,MonthlyCharges=-4); r=c.post("/predict",json=payload,headers={"X-API-Key":"test-secret-key"}); assert r.status_code==422 and "MonthlyCharges" in r.text
def test_invalid_total_charges_returns_clear_422(monkeypatch,sample_customer):
    c=_client(monkeypatch,{"model":FakeModel(),"threshold":.4,"model_version":"v-test","metadata":{}}); payload=dict(sample_customer,TotalCharges="not-a-number"); r=c.post("/predict",json=payload,headers={"X-API-Key":"test-secret-key"}); assert r.status_code==422 and "numeric string" in r.text
def test_batch_prediction(monkeypatch,sample_customer):
    c=_client(monkeypatch,{"model":FakeModel(),"threshold":.4,"model_version":"v-test","metadata":{}}); r=c.post("/predict/batch",json={"customers":[sample_customer,sample_customer]},headers={"X-API-Key":"test-secret-key"}); assert r.status_code==200 and r.json()["count"]==2
def test_batch_prediction_rejects_empty_batch(monkeypatch):
    c=_client(monkeypatch,{"model":FakeModel(),"threshold":.4,"model_version":"v-test","metadata":{}}); assert c.post("/predict/batch",json={"customers":[]},headers={"X-API-Key":"test-secret-key"}).status_code==422
def test_drift_endpoint_returns_psi_results(monkeypatch,sample_customer,synthetic_frame):
    X,_=synthetic_frame; profiles=build_reference_profiles(X); bundle={"model":FakeModel(),"threshold":.4,"model_version":"v-test","metadata":{"drift_reference":profiles}}; c=_client(monkeypatch,bundle); r=c.post("/drift/check",json={"recent_customers":[sample_customer for _ in range(20)]},headers={"X-API-Key":"test-secret-key"}); assert r.status_code==200 and r.json()["sample_size"]==20 and "MonthlyCharges" in r.json()["features"]
def test_health_is_not_ready_when_artifact_missing(monkeypatch):
    def missing():raise FileNotFoundError("train first")
    monkeypatch.setattr("app.main.get_bundle",missing); assert TestClient(app).get("/health").status_code==503
def test_non_finite_charge_values_are_rejected(monkeypatch,sample_customer):
    c=_client(monkeypatch,{"model":FakeModel(),"threshold":.4,"model_version":"v-test","metadata":{}}); payload=dict(sample_customer,MonthlyCharges="NaN"); r=c.post("/predict",json=payload,headers={"X-API-Key":"test-secret-key"}); assert r.status_code==422 and "finite number" in r.text
