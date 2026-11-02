#!/usr/bin/env python3
"""Export FastAPI's generated OpenAPI schema to openapi.json."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.main import app
output=Path(__file__).resolve().parents[1]/"openapi.json"
output.write_text(json.dumps(app.openapi(),indent=2),encoding="utf-8")
print(f"Wrote {output}")
