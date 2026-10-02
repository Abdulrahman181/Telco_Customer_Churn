# Telecom Customer Churn Prediction

This repository contains a Jupyter notebook for churn analysis/modeling and a Streamlit interface (`main.py`) that was written to load pre-trained encoders, a scaler, and classification models.

## Repository status and limitations

The checked-in snapshot does **not** contain the customer dataset, a dependency lock/requirements file, or the `Deployment_Churn` model-artifact directory referenced by the app. The app therefore cannot make predictions from this checkout. It now reports the missing model assets instead of failing with a file-not-found traceback. No model results or performance metrics have been validated as part of this repository snapshot.

The notebook may require an external dataset and a compatible Python environment. Obtain data through an authorized source and verify the notebook's input paths and dependencies before running it. This repository does not claim a reproducible environment because the training dependency versions and saved model files are not present.

## Enabling the Streamlit app

To enable inference, provide the complete artifact set at the paths expected by `main.py`:

- `Deployment_Churn/Encoding/InternetService_le.pkl`
- `Deployment_Churn/Encoding/PaymentMethod_le.pkl`
- `Deployment_Churn/Encoding/tenure_group_le.pkl`
- `Deployment_Churn/Encoding/Contract_oe.pkl`
- `Deployment_Churn/Scaling/scaler.pkl`
- `Deployment_Churn/Models/` containing the nine model files named in `main.py`

Install the exact compatible Python packages used to create those artifacts before starting the app with `streamlit run main.py`. The required versions cannot be inferred safely from the files currently available. **Only load pickle artifacts from a trusted, verified source**: deserializing an untrusted pickle can execute arbitrary code. Keep private data and model artifacts out of public commits unless their disclosure is intentional and approved.

## Customer-input handling

The interface uses the attributes entered in a prediction request to display predictions. It does not retain a history of customer inputs or provide a button to display accumulated records. This is not a substitute for reviewing the privacy and security configuration of any public deployment.
