"""Pydantic contracts for CRM prediction and drift requests."""
from __future__ import annotations
import math
from typing import Literal
from pydantic import BaseModel,ConfigDict,Field,field_validator

class CustomerFeatures(BaseModel):
    """One CRM record; unknown category strings are allowed intentionally."""
    model_config=ConfigDict(extra="forbid")
    gender:str=Field(min_length=1,max_length=80); SeniorCitizen:Literal[0,1]; Partner:str=Field(min_length=1,max_length=80); Dependents:str=Field(min_length=1,max_length=80)
    tenure:int=Field(ge=0,le=100); PhoneService:str=Field(min_length=1,max_length=80); MultipleLines:str=Field(min_length=1,max_length=80); InternetService:str=Field(min_length=1,max_length=80)
    OnlineSecurity:str=Field(min_length=1,max_length=80); OnlineBackup:str=Field(min_length=1,max_length=80); DeviceProtection:str=Field(min_length=1,max_length=80); TechSupport:str=Field(min_length=1,max_length=80)
    StreamingTV:str=Field(min_length=1,max_length=80); StreamingMovies:str=Field(min_length=1,max_length=80); Contract:str=Field(min_length=1,max_length=80); PaperlessBilling:str=Field(min_length=1,max_length=80)
    PaymentMethod:str=Field(min_length=1,max_length=120); MonthlyCharges:float=Field(ge=0,le=100000); TotalCharges:str=Field(max_length=40)
    @field_validator("TotalCharges",mode="before")
    @classmethod
    def validate_total_charges(cls,value):
        if value is None:return ""
        value=str(value).strip()
        if value=="":return ""
        try:numeric=float(value)
        except (TypeError,ValueError) as exc:raise ValueError("TotalCharges must be a numeric string or an empty string") from exc
        if not math.isfinite(numeric):raise ValueError("TotalCharges must be a finite numeric string")
        if numeric<0:raise ValueError("TotalCharges must be greater than or equal to zero")
        return value
    @field_validator("MonthlyCharges",mode="before")
    @classmethod
    def validate_monthly_charges(cls,value):
        try:numeric=float(value)
        except (TypeError,ValueError) as exc:raise ValueError("MonthlyCharges must be numeric") from exc
        if not math.isfinite(numeric):raise ValueError("MonthlyCharges must be a finite number")
        return numeric

class BatchPredictionRequest(BaseModel):
    model_config=ConfigDict(extra="forbid")
    customers:list[CustomerFeatures]=Field(min_length=1,max_length=100)
class DriftCheckRequest(BaseModel):
    model_config=ConfigDict(extra="forbid")
    recent_customers:list[CustomerFeatures]=Field(min_length=20,max_length=1000)
class PredictionResponse(BaseModel):
    churn_probability:float; churn:bool; threshold:float; model_version:str; recommendation:str
class BatchPredictionResponse(BaseModel):
    predictions:list[PredictionResponse]; count:int; model_version:str
