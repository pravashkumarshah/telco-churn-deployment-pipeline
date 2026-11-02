from app.config import FEATURE_COLUMNS
from app.training import load_xy,train_model
def test_load_xy_maps_target_and_excludes_customer_id(tmp_path,synthetic_frame):
 X,y=synthetic_frame; frame=X.copy(); frame.insert(0,"customerID",[f"C{i:05d}" for i in range(len(X))]); frame["Churn"]=y.map({0:"No",1:"Yes"}); path=tmp_path/"source.csv"; frame.to_csv(path,index=False); loaded,labels,_=load_xy(path); assert list(loaded.columns)==FEATURE_COLUMNS and set(labels.unique())<={0,1}
def test_load_xy_rejects_duplicate_customer_ids(tmp_path,synthetic_frame):
 X,y=synthetic_frame; frame=X.copy(); frame.insert(0,"customerID",["same"]*len(X)); frame["Churn"]=y.map({0:"No",1:"Yes"}); path=tmp_path/"duplicate.csv"; frame.to_csv(path,index=False)
 try:load_xy(path)
 except ValueError as exc:assert "Duplicate customerID" in str(exc)
 else:raise AssertionError("duplicate IDs must be rejected")
def test_training_writes_versioned_bundle_and_metadata(tmp_path,synthetic_frame):
 X,y=synthetic_frame; frame=X.copy(); frame.insert(0,"customerID",[f"C{i:05d}" for i in range(len(X))]); frame["Churn"]=y.map({0:"No",1:"Yes"}); path=tmp_path/"synthetic_telco.csv"; frame.to_csv(path,index=False); output=tmp_path/"artifacts"; result=train_model(path,output,n_iter=1,cv_folds=2,random_state=11); assert (output/"model_bundle.joblib").exists() and (output/"metadata.json").exists() and result["metadata"]["training_data_sha256"] and 0<=result["metadata"]["threshold_selection"]["threshold"]<=1
