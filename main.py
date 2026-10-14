"""Streamlit application for the Telco customer-churn model artifacts.

The repository intentionally does not contain trained models or encoders.  The
application therefore validates the artifact directory before rendering a
prediction form and never silently creates or downloads replacements.
"""

from __future__ import annotations

import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:  # Keep static checks and the artifact validator usable without Streamlit.
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover - exercised in deployment only.
    st = None  # type: ignore[assignment]


ARTIFACT_ROOT = Path("Deployment_Churn")
ENCODER_FILES = {
    "internet_service": Path("Encoding/InternetService_le.pkl"),
    "payment_method": Path("Encoding/PaymentMethod_le.pkl"),
    "contract": Path("Encoding/Contract_oe.pkl"),
}
SCALER_FILE = Path("Scaling/scaler.pkl")
MODEL_FILES = {
    "Decision Tree": Path("Models/Decision Tree.pkl"),
    "Logistic Regression": Path("Models/LogisticRegression.pkl"),
    "SVC": Path("Models/SVC.pkl"),
    "KNN": Path("Models/KNN.pkl"),
    "Random Forest": Path("Models/RandomForest.pkl"),
    "Gradient Boosting": Path("Models/GradientBoosting.pkl"),
    "XGBoost": Path("Models/XGBoost.pkl"),
    "AdaBoost": Path("Models/AdaBoost.pkl"),
    "Stacking": Path("Models/Stacking.pkl"),
}

# This is the feature order used by the original training notebook after it
# drops gender and scales the three numeric columns at the end.
class ArtifactBundle(dict[str, Any]):
    """Loaded encoders, scaler, and models with a predictable mapping."""


def required_artifacts(root: Path) -> list[Path]:
    """Return every file needed by the dashboard, relative to ``root``."""
    relative = list(ENCODER_FILES.values()) + [SCALER_FILE] + list(MODEL_FILES.values())
    return [root / ARTIFACT_ROOT / path for path in relative]


def missing_artifacts(root: Path | None = None) -> list[Path]:
    """Return missing artifact paths without attempting to load any pickle."""
    project_root = (root or Path(__file__).resolve().parent).resolve()
    return [path for path in required_artifacts(project_root) if not path.is_file()]


def _load_pickle(path: Path) -> Any:
    """Load a repository-local pickle using a context manager.

    Pickle is executable serialization.  Only load files supplied by a trusted
    training pipeline; this function deliberately performs no network fetch.
    """
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_artifacts(root: Path | None = None) -> ArtifactBundle:
    """Load all model artifacts or raise an actionable error.

    No fallback model or synthetic artifact is created when files are absent.
    """
    project_root = (root or Path(__file__).resolve().parent).resolve()
    missing = missing_artifacts(project_root)
    if missing:
        listed = "\n".join(f"- {path.relative_to(project_root)}" for path in missing)
        raise FileNotFoundError(
            "Required model artifacts are missing. Train/export the model first "
            "and place the trusted files under Deployment_Churn/; missing files:\n"
            f"{listed}"
        )

    artifact_root = project_root / ARTIFACT_ROOT
    encoders = {
        name: _load_pickle(artifact_root / relative)
        for name, relative in ENCODER_FILES.items()
    }
    scaler = _load_pickle(artifact_root / SCALER_FILE)
    models = {
        name: _load_pickle(artifact_root / relative)
        for name, relative in MODEL_FILES.items()
    }
    return ArtifactBundle(encoders=encoders, scaler=scaler, models=models)


def _yes_no(value: str) -> int:
    return int(value == "Yes")


def build_input(choices: dict[str, Any], artifacts: ArtifactBundle) -> np.ndarray:
    """Encode one UI record in the training feature order."""
    encoders = artifacts["encoders"]
    # Match the notebook's post-drop column order exactly.  Gender is not in
    # this list because the training notebook drops it before fitting.
    encoded = [
        _yes_no(str(choices["SeniorCitizen"])),
        _yes_no(str(choices["Partner"])),
        _yes_no(str(choices["Dependents"])),
        _yes_no(str(choices["PhoneService"])),
        _yes_no(str(choices["MultipleLines"])),
        int(encoders["internet_service"].transform([choices["InternetService"]])[0]),
        _yes_no(str(choices["OnlineSecurity"])),
        _yes_no(str(choices["OnlineBackup"])),
        _yes_no(str(choices["DeviceProtection"])),
        _yes_no(str(choices["TechSupport"])),
        _yes_no(str(choices["StreamingTV"])),
        _yes_no(str(choices["StreamingMovies"])),
        int(encoders["contract"].transform([[choices["Contract"]]])[0][0]),
        _yes_no(str(choices["PaperlessBilling"])),
        int(encoders["payment_method"].transform([choices["PaymentMethod"]])[0]),
    ]
    # PaymentMethod is a categorical feature in the original 18-column model;
    # keep it adjacent to the other encoded categorical values and do not treat
    # its label as a binary value.
    encoded[-1] = int(encoders["payment_method"].transform([choices["PaymentMethod"]])[0])
    numeric = np.asarray(
        [[choices["MonthlyCharges"], choices["TotalCharges"], choices["tenure"]]],
        dtype=float,
    )
    scaled = artifacts["scaler"].transform(numeric)
    return np.concatenate((np.asarray([encoded], dtype=float), scaled), axis=1)


def predict_churn(models: dict[str, Any], input_data: np.ndarray) -> dict[str, str]:
    """Return one deterministic class label per model."""
    if input_data.ndim != 2 or input_data.shape[0] != 1:
        raise ValueError("input_data must be a 2-D array containing exactly one record")
    predictions: dict[str, str] = {}
    for name, model in models.items():
        prediction = int(model.predict(input_data)[0])
        predictions[name] = "Churn" if prediction == 1 else "Not Churn"
    return predictions


def save_user_data(choices: dict[str, Any], prediction: str) -> None:
    """Store only the current session's submitted values in Streamlit state."""
    if st is None:
        return
    record = {**choices, "Prediction": prediction}
    st.session_state.setdefault("user_data_list", []).append(record)


def display_predictions(predictions: dict[str, str]) -> None:
    """Display individual predictions and a deterministic majority status."""
    if st is None:
        return
    st.subheader("Prediction results")
    st.dataframe(
        pd.DataFrame(
            [{"Model": model, "Prediction": result} for model, result in predictions.items()]
        ),
        hide_index=True,
        use_container_width=True,
    )
    majority = Counter(predictions.values()).most_common(1)[0][0]
    st.success(f"Customer status: {majority}")


def _render_form(artifacts: ArtifactBundle) -> dict[str, Any]:
    """Render the validated input form and return its values."""
    if st is None:  # pragma: no cover
        raise RuntimeError("Streamlit is required to render the dashboard")
    encoders = artifacts["encoders"]
    with st.form("customer_attributes"):
        col1, col2, col3 = st.columns(3)
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", ["Yes", "No"])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.number_input(
                "Tenure (months)", min_value=0, max_value=120, value=0, step=1
            )
            phone = st.selectbox("Phone Service", ["Yes", "No"])
        with col2:
            multiple = st.selectbox("Multiple Lines", ["Yes", "No"])
            internet = st.selectbox("Internet Service", list(encoders["internet_service"].classes_))
            security = st.selectbox("Online Security", ["Yes", "No"])
            backup = st.selectbox("Online Backup", ["Yes", "No"])
            device = st.selectbox("Device Protection", ["Yes", "No"])
            support = st.selectbox("Tech Support", ["Yes", "No"])
            tv = st.selectbox("Streaming TV", ["Yes", "No"])
        with col3:
            movies = st.selectbox("Streaming Movies", ["Yes", "No"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox("Payment Method", list(encoders["payment_method"].classes_))
            monthly = st.number_input("Monthly Charges", min_value=0.0, value=0.0)
            total = st.number_input("Total Charges", min_value=0.0, value=0.0)
        submitted = st.form_submit_button("Predict churn")
    if not submitted:
        return {}
    return {
        "Gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": device,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }


def main() -> None:
    """Start the dashboard, stopping cleanly when prerequisites are absent."""
    if st is None:
        raise RuntimeError(
            "Streamlit is not installed. Install requirements.txt and run: "
            "streamlit run main.py"
        )
    st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉", layout="wide")
    st.title("Telecom Customer Churn Prediction")
    st.caption("Predictions use only trusted, locally supplied model artifacts.")
    try:
        artifacts = load_artifacts()
    except (FileNotFoundError, OSError, ValueError, pickle.UnpicklingError) as error:
        st.error(str(error))
        st.info("No prediction was run because the required artifacts are unavailable or invalid.")
        return

    choices = _render_form(artifacts)
    if choices:
        try:
            predictions = predict_churn(artifacts["models"], build_input(choices, artifacts))
        except (KeyError, ValueError, TypeError) as error:
            st.error(f"Input could not be encoded with the supplied artifacts: {error}")
            return
        save_user_data(choices, Counter(predictions.values()).most_common(1)[0][0])
        display_predictions(predictions)

    if st.button("Show submitted records"):
        records = st.session_state.get("user_data_list", [])
        st.dataframe(pd.DataFrame(records), hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()
