# Threshold selection and drift monitoring

## Decision threshold

The classifier emits a churn probability. A threshold converts it into a retention-review flag. The project selects the threshold on the validation split by minimizing `FN * false_negative_cost + FP * false_positive_cost`. The default costs (250 and 40 units) are illustrative placeholders and must be replaced with business-approved estimates. Do not tune the threshold on the final test split.

## Population Stability Index

PSI compares feature distributions in a recent batch with training-reference bins/categories. The service flags a feature when PSI > 0.20. This is a screening trigger, not proof of model degradation. Investigate upstream data changes and validate outcome metrics before retraining. Use a sufficiently large monitoring window and retain a rollback model.
