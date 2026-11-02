#!/usr/bin/env python3
"""Run real performance benchmark for single and batch prediction endpoints."""
import json
import os
import platform
import statistics
import time
import urllib.request
import urllib.error
from pathlib import Path

def benchmark_endpoint(url: str, payload_bytes: bytes, api_key: str, iterations: int = 100, warmup: int = 10):
    req = urllib.request.Request(
        url,
        data=payload_bytes,
        headers={"Content-Type": "application/json", "X-API-Key": api_key},
        method="POST"
    )

    print(f"Starting benchmark for {url} ({warmup} warmup, {iterations} measured iterations)...")
    
    # Warmup
    for _ in range(warmup):
        with urllib.request.urlopen(req, timeout=15) as res:
            res.read()

    latencies = []
    errors = 0

    t_total_start = time.perf_counter()
    for _ in range(iterations):
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=15) as res:
                res.read()
                if res.status == 200:
                    latencies.append((time.perf_counter() - t0) * 1000)
                else:
                    errors += 1
        except Exception as e:
            errors += 1
    t_total_end = time.perf_counter()

    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)

    return {
        "endpoint": url,
        "total_requests": iterations,
        "successful_requests": n,
        "failed_requests": errors,
        "total_duration_sec": round(t_total_end - t_total_start, 3),
        "throughput_req_per_sec": round(n / (t_total_end - t_total_start), 2),
        "min_latency_ms": round(latencies_sorted[0], 2) if n else None,
        "median_latency_ms": round(statistics.median(latencies_sorted), 2) if n else None,
        "mean_latency_ms": round(statistics.mean(latencies_sorted), 2) if n else None,
        "p90_latency_ms": round(latencies_sorted[int(0.90 * n)], 2) if n else None,
        "p95_latency_ms": round(latencies_sorted[int(0.95 * n)], 2) if n else None,
        "p99_latency_ms": round(latencies_sorted[min(n - 1, int(0.99 * n))], 2) if n else None,
        "max_latency_ms": round(latencies_sorted[-1], 2) if n else None
    }

def main():
    api_key = os.getenv("CHURN_API_KEY", "telco-secret-key-churn-2026-secure")
    single_payload = Path("examples/single_prediction.json").read_bytes()
    batch_payload = Path("examples/batch_prediction.json").read_bytes()

    single_results = benchmark_endpoint("http://127.0.0.1:8000/predict", single_payload, api_key, iterations=100, warmup=10)
    batch_results = benchmark_endpoint("http://127.0.0.1:8000/predict/batch", batch_payload, api_key, iterations=100, warmup=10)

    report = {
        "benchmark_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "os": platform.platform(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
            "machine": platform.machine()
        },
        "single_prediction_benchmark": single_results,
        "batch_prediction_benchmark": batch_results
    }

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "performance_benchmark.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n--- Benchmark Results Saved to {out_path} ---")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()

