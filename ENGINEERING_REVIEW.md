# Engineering Review: Telco_Customer_Churn

## Scope and baseline

The audit covered every tracked Python file, the training notebook, README, and repository configuration. The baseline repository contained only `main.py`, `Customer_Churn_Prediction.ipynb`, and `README.md`; it had no `requirements.txt`, tests, dataset, or `Deployment_Churn/` artifacts.

This review deliberately does **not** invent data, model files, metrics, or a successful training result.

## Findings and fixes

### Critical / blocking

1. **The dashboard could not start from a clean checkout.** `main.py` opened nine model pickles and four encoder/scaler files at import time using relative paths, but none of those files were in the repository. It also imported Streamlit unconditionally. The app now loads artifacts only inside `main()`, resolves paths relative to the project, validates every required file, and stops with an explicit missing-artifact message. No fallback or synthetic model is created.
2. **The notebook used a Colab-only path** (`/content/WA_Fn-UseC_-Telco-Customer-Churn.csv`). It now accepts `TELCO_DATA_PATH` or a local filename and fails clearly when the trusted CSV is absent.
3. **Target/data leakage existed in resampling.** The original notebook applied `SMOTE` to the complete feature matrix before making train/validation/test splits. Synthetic observations could therefore influence held-out data and CV folds. The resampling cell was removed; the split is now stratified before any fitting, with the original imbalanced data retained. If resampling is reintroduced, it must be inside an imbalanced-learn pipeline and fitted within each training fold only.
4. **The original split was not stratified.** The train/validation/test split now uses `stratify=y` and fixed `random_state=42` for reproducibility.
5. **The notebook had stale outputs.** It contained 87 stored outputs and execution history from an unavailable environment. All code-cell outputs and execution counts were removed so old charts and scores cannot be mistaken for current results.

### Correctness and maintainability

- Replaced deprecated/incompatible XGBoost and AdaBoost arguments in the notebook (`use_label_encoder`, `SAMME.R`) and set explicit evaluation metrics/random states where appropriate.
- Removed the stray `from re import X` and replaced the overwritten `LabelEncoder` loop with independent encoders.
- Reworked `main.py` into import-safe functions (`load_artifacts`, `build_input`, `predict_churn`, form rendering) so static checks and tests can run without Streamlit or model files installed.
- Preserved the notebook's post-drop feature order in the dashboard: 15 encoded categorical/binary features followed by scaled `MonthlyCharges`, `TotalCharges`, and `tenure` (18 columns total). `gender` remains omitted because the notebook drops it; retraining should publish an explicit schema manifest.
- Added context-managed file loading, input shape validation, fixed model ordering, local-only artifact loading, and deterministic majority display.
- Added `requirements.txt`, `.gitignore`, Ruff/Pytest configuration, and tests for Python parsing, notebook hygiene, and the missing-artifact path.

### Metrics, leakage, and production risks retained for follow-up

- Model selection in the notebook is still primarily accuracy-oriented. Churn recall, precision-recall trade-offs, calibration, threshold selection, confidence intervals, subgroup/fairness checks, and a locked final test protocol require a real dataset and a defined business objective.
- The dashboard loads pickle artifacts. Pickle is executable serialization; only artifacts produced and reviewed by a trusted training process should be supplied. A safer production contract would use a signed/versioned model format plus a schema manifest.
- The dashboard cannot prove encoder/model compatibility from filenames alone. `n_features_in_` and a feature-schema hash should be exported and checked as part of the next training release.
- No secrets were found in the tracked files. The original UI used a remote background image and raw HTML; the revised UI removes that external dependency and uses native Streamlit output.

## Validation performed

- Python AST parsing and bytecode compilation passed for all Python files.
- AST parsing passed for every code cell in the notebook; `ruff --select F821` found no undefined names there.
- `ruff check main.py tests` passed after the refactor.
- `pytest -q` passed: 3 tests.
- Notebook JSON parsing passed; code-cell outputs and execution counts are zero; the Colab `/content/` path is absent.
- `git diff --check` passed.
- Full notebook training and Streamlit execution were **not** run: the repository has no dataset or model artifacts, and the sandbox did not have Streamlit, XGBoost, or `nbformat` installed at audit time. Therefore this report contains no model metrics or deployment claim.

## Changed files

- `main.py`
- `Customer_Churn_Prediction.ipynb`
- `README.md`
- `requirements.txt`
- `.gitignore`
- `pyproject.toml`
- `tests/test_repo_health.py`
- `ENGINEERING_REVIEW.md`
