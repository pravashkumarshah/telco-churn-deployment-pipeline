"""Population Stability Index (PSI) profiles for numeric and categorical inputs."""
from __future__ import annotations
import numpy as np
import pandas as pd
from app.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES
_EPSILON=1e-6
_MISSING="__MISSING__"
_OTHER="__OTHER__"

def build_reference_profiles(X:pd.DataFrame,max_categories:int=30)->dict:
    """Build quantile-bin/category baseline distributions using training data only."""
    profiles={}
    for feature in NUMERIC_FEATURES:
        series=pd.to_numeric(X[feature],errors="coerce"); valid=series.dropna().to_numpy(dtype=float)
        internal=np.array([],dtype=float) if len(valid)==0 else np.unique(np.quantile(valid,np.linspace(.1,.9,9)))
        if len(valid):internal=internal[(internal>valid.min())&(internal<valid.max())]
        edges=np.concatenate(([-np.inf],internal,[np.inf])).tolist()
        profiles[feature]={"kind":"numeric","edges":edges,"reference":_normalise(_numeric_counts(series,edges))}
    for feature in CATEGORICAL_FEATURES:
        values=X[feature].fillna(_MISSING).astype(str); top=values.value_counts().head(max_categories).index.tolist(); keys=list(dict.fromkeys([*top,_OTHER,_MISSING]))
        profiles[feature]={"kind":"categorical","keys":keys,"reference":_normalise(_category_counts(values,keys))}
    return profiles

def calculate_psi(reference_profiles:dict,recent:pd.DataFrame)->dict:
    """Compute PSI per feature; values > 0.20 are flagged."""
    results={}
    for feature,profile in reference_profiles.items():
        if feature not in recent.columns:
            results[feature]={"psi":None,"error":"feature_missing"}; continue
        if profile["kind"]=="numeric":
            values=pd.to_numeric(recent[feature],errors="coerce"); actual=_normalise(_numeric_counts(values,profile["edges"]))
        else:
            values=recent[feature].fillna(_MISSING).astype(str); actual=_normalise(_category_counts(values,profile["keys"]))
        expected=np.asarray(profile["reference"],dtype=float); actual=np.asarray(actual,dtype=float)
        psi=float(np.sum((actual-expected)*np.log((actual+_EPSILON)/(expected+_EPSILON))))
        results[feature]={"psi":psi,"flagged":bool(psi>.20)}
    return results

def _numeric_counts(series:pd.Series,edges:list[float])->list[int]:
    values=pd.to_numeric(series,errors="coerce").to_numpy(dtype=float); missing=np.isnan(values); bins=np.histogram(values[~missing],bins=np.asarray(edges,dtype=float))[0].tolist()
    bins.append(int(missing.sum())); return bins

def _category_counts(values:pd.Series,keys:list[str])->list[int]:
    normal=[key for key in keys if key not in (_OTHER,_MISSING)]
    mapped=values.map(lambda value:value if value in normal or value==_MISSING else _OTHER); counts=mapped.value_counts()
    return [int(counts.get(key,0)) for key in keys]

def _normalise(counts)->list[float]:
    arr=np.asarray(counts,dtype=float)
    if len(arr)==0:return []
    arr=(arr+_EPSILON)/(arr.sum()+_EPSILON*len(arr)); return arr.tolist()
