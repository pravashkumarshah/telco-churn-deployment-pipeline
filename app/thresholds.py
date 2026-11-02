"""Business-cost threshold selection and binary classification metrics."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

def choose_cost_threshold(y_true, probabilities, false_negative_cost: float, false_positive_cost: float, candidates=None) -> dict:
    y=np.asarray(y_true,dtype=int); p=np.asarray(probabilities,dtype=float)
    if y.ndim!=1 or p.ndim!=1 or len(y)!=len(p) or len(y)==0: raise ValueError("y_true and probabilities must be non-empty 1-D arrays of equal length")
    if not np.isin(y,[0,1]).all(): raise ValueError("y_true must contain only 0 and 1")
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any(): raise ValueError("probabilities must be finite values between 0 and 1")
    if false_negative_cost<0 or false_positive_cost<0: raise ValueError("cost values must be non-negative")
    if candidates is None: candidates=np.linspace(0.0,1.0,201)
    best=None
    for threshold in candidates:
        predicted=(p>=float(threshold)).astype(int); tn,fp,fn,tp=confusion_matrix(y,predicted,labels=[0,1]).ravel(); cost=float(fn*false_negative_cost+fp*false_positive_cost)
        candidate={"threshold":float(threshold),"total_cost":cost,"false_negatives":int(fn),"false_positives":int(fp),"true_positives":int(tp),"true_negatives":int(tn)}
        if best is None or (cost,-float(threshold))<(best["total_cost"],-best["threshold"]): best=candidate
    return best

def classification_metrics(y_true, probabilities, threshold: float) -> dict:
    y=np.asarray(y_true,dtype=int); p=np.asarray(probabilities,dtype=float)
    if len(y)!=len(p) or len(y)==0: raise ValueError("y_true and probabilities must have equal non-zero lengths")
    pred=(p>=threshold).astype(int); tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    result={"threshold":float(threshold),"accuracy":float(accuracy_score(y,pred)),"balanced_accuracy":float(balanced_accuracy_score(y,pred)),"precision":float(precision_score(y,pred,zero_division=0)),"recall":float(recall_score(y,pred,zero_division=0)),"f1":float(f1_score(y,pred,zero_division=0)),"confusion_matrix":[[int(tn),int(fp)],[int(fn),int(tp)]],"true_negatives":int(tn),"false_positives":int(fp),"false_negatives":int(fn),"true_positives":int(tp)}
    if len(np.unique(y))==2: result["roc_auc"]=float(roc_auc_score(y,p)); result["average_precision"]=float(average_precision_score(y,p))
    else: result["roc_auc"]=None; result["average_precision"]=None
    return result
