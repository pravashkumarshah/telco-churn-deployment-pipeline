#!/usr/bin/env python3
"""Automated verification script for container endpoints, auth, and health."""
import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("CHURN_API_KEY", "telco-secret-key-churn-2026-secure")

def request(path, method="GET", data=None, headers=None):
    url = f"{BASE_URL}{path}"
    h = headers or {}
    payload = None
    if data is not None:
        payload = json.dumps(data).encode("utf-8") if isinstance(data, (dict, list)) else data
        h["Content-Type"] = "application/json"
    
    req = urllib.request.Request(url, data=payload, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as res:
            body = res.read().decode("utf-8")
            try:
                parsed = json.loads(body)
            except Exception:
                parsed = body
            return res.status, parsed
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        try:
            parsed = json.loads(body)
        except Exception:
            parsed = body
        return e.code, parsed
    except Exception as e:
        return 0, str(e)

def main():
    print(f"--- Verifying Container Endpoints at {BASE_URL} ---")
    results = {}

    # 1. GET / (Frontend serving)
    status, body = request("/")
    is_html = isinstance(body, str) and "<!DOCTYPE html>" in body and "Meridian Telecom" in body
    results["frontend_get_root"] = {
        "status": status,
        "valid_html": is_html,
        "pass": status == 200 and is_html
    }
    print(f"1. GET / -> Status {status}, Valid HTML: {is_html}")

    # 2. GET /health
    status, body = request("/health")
    is_healthy = isinstance(body, dict) and body.get("status") == "ok" and "model_version" in body
    results["health_check"] = {
        "status": status,
        "body": body,
        "pass": status == 200 and is_healthy
    }
    print(f"2. GET /health -> Status {status}, Health OK: {is_healthy}")

    # 3. Auth Tests: Missing key, Wrong key, Valid key on POST /predict
    single_payload = json.loads(Path("examples/single_prediction.json").read_text(encoding="utf-8"))

    # Missing Key
    status, body = request("/predict", method="POST", data=single_payload)
    results["auth_missing_key"] = {"status": status, "pass": status == 401}
    print(f"3a. POST /predict (No Key) -> Status {status} (Expected 401)")

    # Wrong Key
    status, body = request("/predict", method="POST", data=single_payload, headers={"X-API-Key": "wrong-secret-key"})
    results["auth_wrong_key"] = {"status": status, "pass": status == 401}
    print(f"3b. POST /predict (Wrong Key) -> Status {status} (Expected 401)")

    # Valid Key
    status, body = request("/predict", method="POST", data=single_payload, headers={"X-API-Key": API_KEY})
    is_valid_pred = isinstance(body, dict) and "churn_probability" in body and "threshold" in body
    results["single_prediction"] = {
        "status": status,
        "body": body,
        "pass": status == 200 and is_valid_pred
    }
    print(f"3c. POST /predict (Valid Key) -> Status {status}, Churn: {body.get('churn') if is_valid_pred else 'N/A'}")

    # 4. POST /predict/batch
    batch_payload = json.loads(Path("examples/batch_prediction.json").read_text(encoding="utf-8"))
    status, body = request("/predict/batch", method="POST", data=batch_payload, headers={"X-API-Key": API_KEY})
    is_valid_batch = isinstance(body, dict) and "predictions" in body and len(body["predictions"]) == 2
    results["batch_prediction"] = {
        "status": status,
        "count": len(body.get("predictions", [])) if isinstance(body, dict) else 0,
        "pass": status == 200 and is_valid_batch
    }
    print(f"4. POST /predict/batch -> Status {status}, Count: {results['batch_prediction']['count']}")

    # 5. POST /drift/check
    drift_customers = [single_payload for _ in range(25)]
    drift_payload = {"recent_customers": drift_customers}
    status, body = request("/drift/check", method="POST", data=drift_payload, headers={"X-API-Key": API_KEY})
    is_valid_drift = isinstance(body, dict) and "features" in body and "psi_alert_threshold" in body
    results["drift_check"] = {
        "status": status,
        "features_count": len(body.get("features", {})) if isinstance(body, dict) else 0,
        "pass": status == 200 and is_valid_drift
    }
    print(f"5. POST /drift/check -> Status {status}, Features: {results['drift_check']['features_count']}")

    # Overall outcome
    all_passed = all(v["pass"] for v in results.values())
    print("\n--- Summary ---")
    for k, v in results.items():
        print(f"  {k}: {'PASS' if v['pass'] else 'FAIL'}")
    print(f"\nFinal Result: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    if not all_passed:
        sys.exit(1)

if __name__ == "__main__":
    main()
