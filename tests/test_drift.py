from app.config import FEATURE_COLUMNS
from app.drift import build_reference_profiles,calculate_psi
def test_same_data_has_low_psi(synthetic_frame):
 X,_=synthetic_frame; ref=build_reference_profiles(X); result=calculate_psi(ref,X); assert max(v["psi"] for v in result.values())<.01
def test_shifted_numeric_feature_is_flagged(synthetic_frame):
 X,_=synthetic_frame; ref=build_reference_profiles(X); changed=X.copy(); changed["MonthlyCharges"]=10000.; result=calculate_psi(ref,changed); assert result["MonthlyCharges"]["psi"]>.2 and result["MonthlyCharges"]["flagged"] is True
def test_unknown_category_is_counted_as_other(synthetic_frame):
 X,_=synthetic_frame; ref=build_reference_profiles(X); changed=X.copy(); changed["PaymentMethod"]="new-provider-category"; assert calculate_psi(ref,changed)["PaymentMethod"]["psi"]>0
def test_missing_drift_column_returns_error(synthetic_frame):
 X,_=synthetic_frame; ref=build_reference_profiles(X); assert calculate_psi(ref,X.drop(columns=[FEATURE_COLUMNS[0]]))[FEATURE_COLUMNS[0]]["error"]=="feature_missing"
