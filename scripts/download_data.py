#!/usr/bin/env python3
"""Download the public IBM sample and verify its expected SHA-256 checksum."""
from __future__ import annotations
import argparse, hashlib, urllib.request, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import DATA_URL, EXPECTED_DATA_SHA256

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/raw/Telco-Customer-Churn.csv")
    args = parser.parse_args()
    destination = Path(args.output); destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {DATA_URL}")
    try:
        urllib.request.urlretrieve(DATA_URL, temporary)
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        if digest != EXPECTED_DATA_SHA256:
            raise RuntimeError(f"Checksum mismatch: expected {EXPECTED_DATA_SHA256}, got {digest}")
        temporary.replace(destination)
        print(f"Saved {destination} (SHA-256: {digest})")
    finally:
        if temporary.exists(): temporary.unlink()

if __name__ == "__main__": main()
