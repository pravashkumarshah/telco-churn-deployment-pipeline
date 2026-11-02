import numpy as np,pandas as pd,pytest
from app.config import FEATURE_COLUMNS
from app.preprocessing import CustomerDataCleaner,build_pipeline
def test_cleaner_converts_total_charges_and_handles_blank(synthetic_frame):
 X,_=synthetic_frame; X=X.copy(); X.loc[X.index[0],"TotalCharges"]=""; out=CustomerDataCleaner().fit_transform(X); assert pd.isna(out.loc[0,"TotalCharges"]) and out["TotalCharges"].dtype.kind in "fi"
def test_cleaner_rejects_missing_feature_column(synthetic_frame):
 X,_=synthetic_frame
 with pytest.raises(ValueError,match="Missing required feature columns"):CustomerDataCleaner().fit_transform(X.drop(columns=[FEATURE_COLUMNS[0]]))
def test_pipeline_fits_and_returns_probabilities(synthetic_frame):
 X,y=synthetic_frame; model=build_pipeline().fit(X,y); p=model.predict_proba(X.iloc[:8]); assert p.shape==(8,2) and np.allclose(p.sum(axis=1),1.)
def test_pipeline_handles_unseen_categories(synthetic_frame):
 X,y=synthetic_frame; model=build_pipeline().fit(X,y); unseen=X.iloc[[0]].copy(); unseen.loc[:,"PaymentMethod"]="Brand new CRM payment category"; assert model.predict_proba(unseen).shape==(1,2)
