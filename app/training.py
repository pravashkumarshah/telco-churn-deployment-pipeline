"""Reproducible training, cross-validated tuning and artefact versioning."""
from __future__ import annotations
import hashlib, json, platform
from datetime import datetime, timezone
from pathlib import Path
import joblib, numpy as np, pandas as pd, sklearn
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from app.config import CATEGORICAL_FEATURES, DATA_URL, FALSE_NEGATIVE_COST, FALSE_POSITIVE_COST, FEATURE_COLUMNS, ID_COLUMN, NUMERIC_FEATURES, TARGET_COLUMN
from app.drift import build_reference_profiles
from app.preprocessing import build_pipeline
from app.thresholds import choose_cost_threshold, classification_metrics

def sha256_file(path: str|Path)->str:
    digest=hashlib.sha256()
    with open(path,"rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()

def sha256_partition(X:pd.DataFrame,y:pd.Series)->str:
    part=X.copy(); part["__target__"]=np.asarray(y,dtype=int)
    return hashlib.sha256(part.to_csv(index=True,lineterminator="\n").encode("utf-8")).hexdigest()

def load_xy(data_path:str|Path):
    frame=pd.read_csv(data_path); required={ID_COLUMN,TARGET_COLUMN,*FEATURE_COLUMNS}; missing=sorted(required-set(frame.columns))
    if missing: raise ValueError(f"Dataset is missing required columns: {missing}")
    if frame[ID_COLUMN].duplicated().any(): raise ValueError("Duplicate customerID values detected; resolve source duplicates before training")
    y=frame[TARGET_COLUMN].map({"No":0,"Yes":1})
    if y.isna().any(): raise ValueError(f"Unexpected target labels: {sorted(frame.loc[y.isna(),TARGET_COLUMN].astype(str).unique().tolist())}; expected Yes/No")
    return frame.loc[:,FEATURE_COLUMNS].copy(),y.astype(int),frame

def train_model(data_path:str|Path,output_dir:str|Path="artifacts",n_iter:int=12,cv_folds:int=3,random_state:int=42)->dict:
    X,y,raw=load_xy(data_path); source_hash=sha256_file(data_path)
    X_trainval,X_test,y_trainval,y_test=train_test_split(X,y,test_size=.20,random_state=random_state,stratify=y)
    X_train,X_validation,y_train,y_validation=train_test_split(X_trainval,y_trainval,test_size=.25,random_state=random_state,stratify=y_trainval)
    pipeline=build_pipeline(random_state)
    params={"classifier__n_estimators":[200,300,400],"classifier__max_depth":[None,8,12,18],"classifier__min_samples_split":[2,5,10],"classifier__min_samples_leaf":[1,2,4],"classifier__max_features":["sqrt",.7],"classifier__class_weight":["balanced","balanced_subsample"]}
    cv=StratifiedKFold(n_splits=cv_folds,shuffle=True,random_state=random_state)
    search=RandomizedSearchCV(pipeline,params,n_iter=n_iter,scoring="average_precision",cv=cv,refit=True,random_state=random_state,n_jobs=-1,verbose=1,return_train_score=True,error_score="raise")
    search.fit(X_train,y_train); model=search.best_estimator_
    train_hash=sha256_partition(X_train,y_train); val_hash=sha256_partition(X_validation,y_validation); test_hash=sha256_partition(X_test,y_test)
    val_prob=model.predict_proba(X_validation)[:,1]
    threshold=choose_cost_threshold(y_validation,val_prob,FALSE_NEGATIVE_COST,FALSE_POSITIVE_COST); selected=threshold["threshold"]
    test_prob=model.predict_proba(X_test)[:,1]
    val_metrics=classification_metrics(y_validation,val_prob,selected); test_metrics=classification_metrics(y_test,test_prob,selected)
    drift=build_reference_profiles(X_train); timestamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"); version=f"telco-churn-{timestamp}-{train_hash[:8]}"
    metadata={"model_version":version,"created_at_utc":datetime.now(timezone.utc).isoformat(),"dataset_name":"IBM Telco Customer Churn sample","dataset_url":DATA_URL,"source_dataset_sha256":source_hash,"training_data_sha256":train_hash,"validation_data_sha256":val_hash,"test_data_sha256":test_hash,"row_count":int(len(raw)),"feature_columns":FEATURE_COLUMNS,"numeric_features":NUMERIC_FEATURES,"categorical_features":CATEGORICAL_FEATURES,"target_column":TARGET_COLUMN,"target_mapping":{"No":0,"Yes":1},"split_sizes":{"train":len(X_train),"validation":len(X_validation),"test":len(X_test)},"cross_validation":{"strategy":"StratifiedKFold","folds":cv_folds,"scoring":"average_precision","best_cv_average_precision":float(search.best_score_)},"best_params":search.best_params_,"threshold_selection":{**threshold,"false_negative_cost":FALSE_NEGATIVE_COST,"false_positive_cost":FALSE_POSITIVE_COST,"selection_data":"validation_only"},"validation_metrics":val_metrics,"test_metrics":test_metrics,"library_versions":{"python":platform.python_version(),"pandas":pd.__version__,"numpy":np.__version__,"scikit_learn":sklearn.__version__,"joblib":joblib.__version__},"drift_reference":drift,"limitations":["Cross-sectional sample; random stratified validation is not temporal backtesting.","Cost values are illustrative and require business approval.","Predictive risk does not establish that a retention offer causes a customer to stay."]}
    output=Path(output_dir); output.mkdir(parents=True,exist_ok=True); bundle={"model":model,"threshold":selected,"model_version":version,"metadata":metadata}
    joblib.dump(bundle,output/"model_bundle.joblib",compress=3)
    (output/"metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    (output/"test_metrics.json").write_text(json.dumps(test_metrics,indent=2),encoding="utf-8")
    (output/"experiment_summary.json").write_text(json.dumps({k:v for k,v in metadata.items() if k!="drift_reference"},indent=2),encoding="utf-8")
    return {"model_path":str(output/"model_bundle.joblib"),"metadata":metadata}
