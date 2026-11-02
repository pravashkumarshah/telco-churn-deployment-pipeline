# Model Card: Telecom Customer Churn Classifier

## Identity and intended use
- Model family: scikit-learn `RandomForestClassifier` in a serializable pipeline.
- Task: binary churn classification (`Yes`/`No`).
- Intended use: prioritize customers for human-reviewed retention outreach. Not for automated account closure, eligibility denial, or price setting.
- Status: educational candidate. No real-data performance scores are claimed until training is run on the verified public CSV.

## Data and features
IBM Telco Customer Churn sample, 7,043 rows and 21 raw columns. `customerID` is excluded. `TotalCharges` is parsed from text; blanks are imputed in the fitted pipeline. Numeric fields: `SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`. Categorical fields cover demographics, services, contract, billing and payment method. Dataset is a cross-sectional snapshot, not a monthly panel.

## Training and evaluation
Stratified 60/20/20 train/validation/test split. RandomizedSearchCV with StratifiedKFold optimizes average precision. Threshold is selected by validation business cost only; test split is held out for final scoring. Metadata records source hash, partition hashes, random seed, library versions, parameters and metrics.

## Costs and threshold
Illustrative assumptions: false negative = 250 cost units; false positive = 40. Minimize `FN*250 + FP*40` on validation. Replace with business-approved expected contribution, offer and contact costs. Probability does not estimate causal retention uplift.

## Limitations and risks
Public sample may not represent a carrier or current population. Random splitting is not temporal backtesting. Historical churn risk does not prove that an offer prevents churn. Review segment performance and potential proxy effects. PSI detects covariate shift, not model performance decline.

## Monitoring and security
PSI > 0.20 is a review trigger; assess data quality and delayed-label metrics before retraining. Monitor recall, precision, PR-AUC, calibration, error cost and segment metrics. API key comes from `CHURN_API_KEY`; use secret management and TLS. Do not log API keys, raw customer payloads, or commit real customer data.
