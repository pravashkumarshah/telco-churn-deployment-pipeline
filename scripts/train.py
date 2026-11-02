#!/usr/bin/env python3
"""CLI entry point for reproducible churn model training."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.training import train_model

def main():
    parser=argparse.ArgumentParser(description="Train and version the telecom churn model")
    parser.add_argument("--data",default="data/raw/Telco-Customer-Churn.csv"); parser.add_argument("--output-dir",default="artifacts"); parser.add_argument("--n-iter",type=int,default=12); parser.add_argument("--cv-folds",type=int,default=3); parser.add_argument("--random-state",type=int,default=42)
    args=parser.parse_args(); result=train_model(args.data,args.output_dir,n_iter=args.n_iter,cv_folds=args.cv_folds,random_state=args.random_state)
    print(json.dumps({"model_path":result["model_path"],"model_version":result["metadata"]["model_version"],"threshold":result["metadata"]["threshold_selection"],"test_metrics":result["metadata"]["test_metrics"]},indent=2))
if __name__=="__main__":main()
