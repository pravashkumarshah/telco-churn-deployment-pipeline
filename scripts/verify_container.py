#!/usr/bin/env python3
"""Verify a running churn API without embedding credentials in source."""
import json
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path

BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("CHURN_API_KEY")


def request(path, method="GET", data=None, headers=None):
    url = f"{BASE_URL}{path}"
    request_headers = dict(headers or {})
    payload = None
    if data is not None:
        payload = json.dumps(data).encode("utf-8")
        request_headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=payload, headers=request_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            body = response.read().decode("utf-8")
            try:
                return response.status, json.loads(body)
            except json.JSONDecodeError:
                return response.status, body
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        try:
            return exc.code, json.loads(body)
        except json.JSONDecodeError:
            return exc.code, body
    except Exception as exc:
        return 0, str(exc)


def main():
    if not API_KEY:
        print("ERROR: Set CHURN_API_KEY in the environment; no default key is provided.", file=sys.stderr)
        sys.exit(2)

    print(f"--- Verifying API at {BASE_URL} ---")
    results = {}

    status, body = request("/")
    valid_html = isinstance(body, str) and "<!DOCTYPE html>" in body and "Meridian Telecom" in body
    results["frontend_get_root"] = status == 200 and valid_html
    print(f"1. GET / -> {status}, valid HTML: {valid_html}")

    status, body = request("/health")
    healthy = isinstance(body, dict) and body.get("status") == "ok" and "model_version" in body
    results["health_check"] = status == 200 and healthy
    print(f"2. GET /health -> {status}, healthy: {healthy}")

    single = json.loads(Path("examples/single_prediction.json").read_text(encoding="utf-8"))
    status, _ = request("/predict", "POST", single)
    results["auth_missing_key"] = status == 401
    print(f"3a. Missing API key -> {status} (expected 401)")

    status, _ = request("/predict", "POST", single, {"X-API-Key": "invalid-test-key"})
    results["auth_wrong_key"] = status == 401
    print(f"3b. Invalid API key -> {status} (expected 401)")

    auth = {"X-API-Key": API_KEY}
    status, body = request("/predict", "POST", single, auth)
    valid_prediction = isinstance(body, dict) and "churn_probability" in body and "threshold" in body
    results["single_prediction"] = status == 200 and valid_prediction
    print(f"3c. Valid single prediction -> {status}")

    batch = json.loads(Path("examples/batch_prediction.json").read_text(encoding="utf-8"))
    status, body = request("/predict/batch", "POST", batch, auth)
    valid_batch = isinstance(body, dict) and len(body.get("predictions", [])) == 2
    results["batch_prediction"] = status == 200 and valid_batch
    print(f"4. Batch prediction -> {status}")

    drift = {"recent_customers": [single for _ in range(25)]}
    status, body = request("/drift/check", "POST", drift, auth)
    valid_drift = isinstance(body, dict) and "features" in body and "psi_alert_threshold" in body
    results["drift_check"] = status == 200 and valid_drift
    print(f"5. Drift check -> {status}")

    for name, passed in results.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    all_passed = all(results.values())
    print(f"Final Result: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
