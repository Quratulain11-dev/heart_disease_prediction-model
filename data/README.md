# Heart disease data

`heart_extended.csv` combines unique rows from the project's original
`heart.csv` with the Hungarian, Switzerland, and VA Long Beach cohorts from the
UCI Heart Disease dataset. The three additional cohorts contribute 617 records
before removal of two exact duplicate feature/target rows. The Cleveland
cohort in the UCI archive is not added because it overlaps the project's
existing data.

The UCI records are converted to the app's category codes: chest pain (`cp`)
and ST slope (`slope`) are zero-based; thal values 3, 6, and 7 become 1, 2, and
3; and target values 1-4 are mapped to disease-present (`1`). The original
project data uses different category encodings and the inverse target labels.
Its `cp` codes map as `{0: 3, 1: 1, 2: 2, 3: 0}`, `restecg` as
`{0: 2, 1: 0, 2: 1}`, `slope` as `{0: 2, 1: 1, 2: 0}`, and `thal` as
`{0: missing, 1: 2, 2: 1, 3: 3}`. Its target is inverted. These mappings were
checked against matching clinical records in the UCI Cleveland cohort.
Unknown values remain blank and are imputed by the training pipeline. The
impossible cholesterol value of zero is treated as missing. The resulting CSV
has 917 unique rows, including 615 from the added cohorts.
Run `python train_model.py` from the project root to tune and retrain the
backend model.

Source: Detrano et al., UCI Machine Learning Repository, *Heart Disease*,
https://doi.org/10.24432/C52P4X

This is a small, historical dataset for educational experimentation, not a
representative clinical population. Nested cross-validation leaves out one
cohort at a time during both model selection and evaluation to reduce leakage
and selection bias. Its scores are not clinical validation and do not establish
that the model is safe for diagnosis or treatment.

The current logistic-regression training run achieved 78.7% nested
leave-one-cohort-out accuracy (balanced accuracy 78.5%, ROC-AUC 0.846), versus
55.3% for predicting the majority class. Accuracy varied by held-out cohort:
77.8% on the existing Cleveland-derived data, 84.0% on Hungary, 72.4% on
Switzerland, and 76.4% on VA Long Beach. These are small, historical cohorts;
the scores should not be interpreted as expected accuracy for current patients.
The model has no training examples older than 77, so predictions for older
patients (such as age 81) are extrapolations and are not validated by these
metrics.
