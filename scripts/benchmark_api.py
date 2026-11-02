#!/usr/bin/env python3
"""Measure end-to-end single-prediction latency against a running API."""
import argparse,json,statistics,time,urllib.error,urllib.request
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument("--url",default="http://127.0.0.1:8000/predict"); p.add_argument("--api-key",required=True); p.add_argument("--payload",default="examples/single_prediction.json"); p.add_argument("--iterations",type=int,default=1000); a=p.parse_args()
    payload=Path(a.payload).read_bytes(); req=urllib.request.Request(a.url,data=payload,headers={"Content-Type":"application/json","X-API-Key":a.api_key},method="POST"); samples=[]
    for i in range(a.iterations+20):
        start=time.perf_counter()
        try:
            with urllib.request.urlopen(req,timeout=10) as response:
                response.read()
                if response.status!=200: raise RuntimeError(f"Unexpected status: {response.status}")
        except urllib.error.HTTPError as exc: raise RuntimeError(f"API returned HTTP {exc.code}: {exc.read().decode(errors='replace')}") from exc
        if i>=20:samples.append((time.perf_counter()-start)*1000)
    ordered=sorted(samples); p95=ordered[min(len(ordered)-1,int(.95*len(ordered)))]; median=statistics.median(samples)
    print(json.dumps({"measured_requests":len(samples),"warmup_requests":20,"median_latency_ms":round(median,3),"p95_latency_ms":round(p95,3),"median_target_ms":100,"meets_median_target":median<100},indent=2))
if __name__=="__main__":main()
