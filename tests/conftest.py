from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from app.config import CATEGORICAL_FEATURES, FEATURE_COLUMNS

@pytest.fixture
def synthetic_frame():
    rng=np.random.default_rng(123); n=180
    values={"gender":["Female","Male"],"Partner":["Yes","No"],"Dependents":["Yes","No"],"PhoneService":["Yes","No"],"MultipleLines":["Yes","No","No phone service"],"InternetService":["DSL","Fiber optic","No"],"OnlineSecurity":["Yes","No","No internet service"],"OnlineBackup":["Yes","No","No internet service"],"DeviceProtection":["Yes","No","No internet service"],"TechSupport":["Yes","No","No internet service"],"StreamingTV":["Yes","No","No internet service"],"StreamingMovies":["Yes","No","No internet service"],"Contract":["Month-to-month","One year","Two year"],"PaperlessBilling":["Yes","No"],"PaymentMethod":["Electronic check","Mailed check","Bank transfer (automatic)"]}
    data={"SeniorCitizen":rng.integers(0,2,n),"tenure":rng.integers(0,73,n),"MonthlyCharges":rng.uniform(20,120,n)}
    data["TotalCharges"]=[f"{x:.2f}" for x in data["tenure"]*data["MonthlyCharges"]*rng.uniform(.8,1.2,n)]
    for col in CATEGORICAL_FEATURES:data[col]=rng.choice(values[col],n)
    frame=pd.DataFrame(data)[FEATURE_COLUMNS]
    prob=1/(1+np.exp(-(-1.8+.045*(24-data["tenure"])+.9*(np.asarray(data["Contract"])=="Month-to-month"))))
    return frame,pd.Series((rng.random(n)<prob).astype(int))

@pytest.fixture
def sample_customer():
    return {"gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"No","tenure":12,"PhoneService":"Yes","MultipleLines":"No","InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"Yes","DeviceProtection":"No","TechSupport":"No","StreamingTV":"Yes","StreamingMovies":"Yes","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":89.5,"TotalCharges":"1074.0"}
