import numpy as np
import pickle as pkl
import streamlit as st
from collections import Counter
from pathlib import Path

# Resolve artifacts relative to this file so Streamlit works from any working directory.
APP_DIR = Path(__file__).resolve().parent
ARTIFACTS = (
    'Deployment_Churn/Encoding/InternetService_le.pkl',
    'Deployment_Churn/Encoding/PaymentMethod_le.pkl',
    'Deployment_Churn/Encoding/tenure_group_le.pkl',
    'Deployment_Churn/Encoding/Contract_oe.pkl',
    'Deployment_Churn/Scaling/scaler.pkl',
    'Deployment_Churn/Models/Decision Tree.pkl',
    'Deployment_Churn/Models/LogisticRegression.pkl',
    'Deployment_Churn/Models/SVC.pkl',
    'Deployment_Churn/Models/KNN.pkl',
    'Deployment_Churn/Models/RandomForest.pkl',
    'Deployment_Churn/Models/GradientBoosting.pkl',
    'Deployment_Churn/Models/XGBoost.pkl',
    'Deployment_Churn/Models/AdaBoost.pkl',
    'Deployment_Churn/Models/Stacking.pkl',
)

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉", layout="wide")
missing_artifacts = [path for path in ARTIFACTS if not (APP_DIR / path).is_file()]
if missing_artifacts:
    st.error(
        "The trained model artifacts required by this app are not included in this repository. "
        "Provide the complete, trusted Deployment_Churn artifact set to enable predictions."
    )
    st.stop()


def load_artifact(relative_path):
    # Only load trusted artifacts; pickle can execute code during deserialization.
    with (APP_DIR / relative_path).open('rb') as artifact_file:
        return pkl.load(artifact_file)


# Load encoders
Internetservice_le = load_artifact('Deployment_Churn/Encoding/InternetService_le.pkl')
Paymentmethod_le = load_artifact('Deployment_Churn/Encoding/PaymentMethod_le.pkl')
Tenure_le = load_artifact('Deployment_Churn/Encoding/tenure_group_le.pkl')
Contract_Oe = load_artifact('Deployment_Churn/Encoding/Contract_oe.pkl')

# Load scaler and models
scaler = load_artifact('Deployment_Churn/Scaling/scaler.pkl')
models = {
    'Decision Tree': load_artifact('Deployment_Churn/Models/Decision Tree.pkl'),
    'Logistic Regression': load_artifact('Deployment_Churn/Models/LogisticRegression.pkl'),
    'SVC': load_artifact('Deployment_Churn/Models/SVC.pkl'),
    'KNN': load_artifact('Deployment_Churn/Models/KNN.pkl'),
    'Random Forest': load_artifact('Deployment_Churn/Models/RandomForest.pkl'),
    'Gradient Boosting': load_artifact('Deployment_Churn/Models/GradientBoosting.pkl'),
    'XGBoost': load_artifact('Deployment_Churn/Models/XGBoost.pkl'),
    'AdaBoost': load_artifact('Deployment_Churn/Models/AdaBoost.pkl'),
    'Stacking': load_artifact('Deployment_Churn/Models/Stacking.pkl'),
}

# Function to predict churn
def predict_churn(input_data):
    predictions = {}
    for name, model in models.items():
        pred = model.predict(input_data)[0]
        predictions[name] = "Churn" if pred == 1 else "Not Churn"
    return predictions

# Function to display predictions with mode highlighted
def display_predictions(predictions):
    st.markdown("<h2 style='text-align: center; color: #4A90E2;'>⚜️Prediction Results⚜️</h2>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns([1, 1], gap="large")  # Two equal-width columns

    # Split predictions into two parts
    predictions_items = list(predictions.items())
    first_half = predictions_items[:5]
    second_half = predictions_items[5:]

    # Display first half in the first column
    with col1:
        for model, result in first_half:
            st.markdown(
                f"<p style='font-size:15px; text-align: center;'>"
                f"<b>{model}:</b> <span style='color: #FFA500;'>{result}</span>"
                f"</p>", unsafe_allow_html=True
            )

    # Display second half in the second column
    with col2:
        for model, result in second_half:
            st.markdown(
                f"<p style='font-size:15px; text-align: center;'>"
                f"<b>{model}:</b> <span style='color: #FFA500;'>{result}</span>"
                f"</p>", unsafe_allow_html=True
            )

    # Calculate and display mode in larger font
    mode_result = Counter(predictions.values()).most_common(1)[0][0]
    if mode_result == "Not Churn":
        st.markdown(f"""<h1 style='text-align: center;font-size:55px; color: white;'>The Customer Status: <span style='color: #27AE60;'>{mode_result}✅</span></h1>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<h1 style='text-align: center;font-size:55px; color: white;'>The Customer Status: <span style='color: #E74C3C;'>{mode_result}😡</span></h1>""", unsafe_allow_html=True)

# Streamlit UI
st.title(" 📞Telecom Customer Churn Prediction")
st.markdown('Predict whether a customer will churn based on various attributes.')

# Photo
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.pexels.com/photos/19683578/pexels-photo-19683578/free-photo-of-analog-telephone-in-sunlight.jpeg?auto=compress&cs=tinysrgb&w=600");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

# Input Form
st.markdown("---")
st.subheader("📝 Enter Customer Attributes:")

c1, c2, c3 = st.columns(3)
with c1:
    gender = st.selectbox("👥 Gender", options=['Male', 'Female'])
    SeniorCitizen = st.selectbox("🎖️ Senior Citizen", options=['Yes', 'No'])
    Partner = st.selectbox("💍 Partner", options=['Yes', 'No'])
    Dependents = st.selectbox("👶 Dependents", options=['Yes', 'No'])
    tenure = st.number_input("📅 Tenure (months)", min_value=0, value=0)
    PhoneService = st.selectbox("📞 Phone Service", options=['Yes', 'No'])

with c2:
    MultipleLines = st.selectbox("📶 Multiple Lines", options=['Yes', 'No'])
    InternetService = st.selectbox("🌐 Internet Service", options=['DSL', 'Fiber optic', 'No'])
    OnlineSecurity = st.selectbox("🔐 Online Security", options=['Yes', 'No'])
    OnlineBackup = st.selectbox("💾 Online Backup", options=['Yes', 'No'])
    DeviceProtection = st.selectbox("🛡️ Device Protection", options=['Yes', 'No'])
    TechSupport = st.selectbox("🛠️ Tech Support", options=['Yes', 'No'])
    StreamingTV = st.selectbox("📺 Streaming TV", options=['Yes', 'No'])

with c3:
    StreamingMovies = st.selectbox("🎥 Streaming Movies", options=['Yes', 'No'])
    Contract = st.selectbox("📃 Contract", options=['Month-to-month', 'One year', 'Two year'])
    PaperlessBilling = st.selectbox("📨 Paperless Billing", options=['Yes', 'No'])
    PaymentMethod = st.selectbox("💳 Payment Method", Paymentmethod_le.classes_)
    MonthlyCharges = st.number_input("💵 Monthly Charges", min_value=0.0, value=0.0)
    TotalCharges = st.number_input("💰 Total Charges", min_value=0.0, value=0.0)

# Encoding Input
gender_encod = 1 if gender == 'Male' else 0
SeniorCitizen_encod = 1 if SeniorCitizen == 'Yes' else 0
Partner_encod = 1 if Partner == 'Yes' else 0
Dependents_encod = 1 if Dependents == 'Yes' else 0
PhoneService_encod = 1 if PhoneService == 'Yes' else 0
MultipleLines_encod = 1 if MultipleLines == 'Yes' else 0
InternetService_encod = Internetservice_le.transform([InternetService])[0]
OnlineSecurity_encod = 1 if OnlineSecurity == 'Yes' else 0
OnlineBackup_encod = 1 if OnlineBackup == 'Yes' else 0
DeviceProtection_encod = 1 if DeviceProtection == 'Yes' else 0
TechSupport_encod = 1 if TechSupport == 'Yes' else 0
StreamingTV_encod = 1 if StreamingTV == 'Yes' else 0
StreamingMovies_encod = 1 if StreamingMovies == 'Yes' else 0
Contract_encod = Contract_Oe.transform([[Contract]])[0][0]
PaperlessBilling_encod = 1 if PaperlessBilling == 'Yes' else 0
PaymentMethod_encod = Paymentmethod_le.transform([PaymentMethod])[0]

# Create input data array
input_data1 = np.array([[ SeniorCitizen_encod, Partner_encod, Dependents_encod
                        , PhoneService_encod, MultipleLines_encod, InternetService_encod,
                        OnlineSecurity_encod, OnlineBackup_encod, DeviceProtection_encod,
                        TechSupport_encod, StreamingTV_encod, StreamingMovies_encod,
                        Contract_encod, PaperlessBilling_encod, PaymentMethod_encod
                        ]])
input_data2 = np.array([[MonthlyCharges, TotalCharges, tenure]])
# Scale input data
input_data3 = scaler.transform(input_data2)
input_data = np.concatenate((input_data1, input_data3), axis=1)




# Button to trigger prediction and save data
if st.button("🔍 Predict Churn"):
    predictions = predict_churn(input_data)
    # Display the predictions
    display_predictions(predictions)
