# Model Card: Telecom Customer Churn Classifier

## Identity and intended use
- Model family: scikit-learn `RandomForestClassifier` inside a serializable preprocessing pipeline.
- Task: binary churn classification (`Yes` / `No`).
- Intended use: prioritize customers for human-reviewed retention outreach. Not for automated account closure, eligibility denial, or price setting.
- Status: educational project candidate; metrics below are from the recorded holdout evaluation in `reports/model_evaluation.json`.

## Data and features
IBM Telco Customer Churn public sample: 7,043 rows and 21 raw columns. `customerID` is checked for duplicates and excluded from model features. `TotalCharges` is parsed from text; blank values are handled by the fitted preprocessing pipeline. Numeric fields: `SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`. Categorical fields cover demographics, services, contract, billing and payment method.

## Training and evaluation
- Stratified split: 4,225 training, 1,409 validation and 1,409 test records.
- RandomizedSearchCV with StratifiedKFold; average precision is the search objective.
- Threshold selected using validation data only; the final test split is held out for evaluation.
- Source dataset SHA-256, split hashes, random seed, library versions, parameters and metrics are recorded in the versioned training metadata.

## Recorded test metrics
At the validation-selected threshold of 0.32, the held-out test results are:
- ROC-AUC: 0.8426
- PR-AUC / average precision: 0.6507
- Precision: 0.4476
- Recall: 0.8904
- F1-score: 0.5957
- Accuracy: 0.6792
- Confusion matrix: TN 624, FP 411, FN 41, TP 333 (test support: 1,409).

At the default threshold of 0.50, recorded recall is 0.7701 and precision is 0.5275. The 0.32 threshold prioritizes finding more churners at the cost of more false-positive retention reviews.

## Costs and threshold
Illustrative assumptions: false negative = 250 cost units; false positive = 40. The threshold is selected on validation data by minimizing `FN*250 + FP*40`. The recorded test-set cost at threshold 0.32 is 26,690 units, compared with 31,820 at threshold 0.50. These are illustrative units, not verified financial savings. Replace the assumptions with business-approved expected contribution, offer and contact costs. Predicted churn risk does not estimate causal retention uplift.

## Limitations and risks
The public sample may not represent a carrier's current customer population. It is a cross-sectional snapshot; random stratified splitting is not temporal backtesting. Review segment performance and potential proxy effects. PSI detects covariate shift, not model performance decline.

## Monitoring and security
PSI > 0.20 is an investigation trigger, not an automatic retraining instruction. Validate data quality and delayed-label outcomes before retraining. Monitor recall, precision, PR-AUC, calibration, error cost and segment metrics. API key comes from `CHURN_API_KEY`; use secret management and TLS. Never log API keys, raw customer payloads, or commit real customer data.
