import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

MODEL_PATH = "fraud_detection_pipeline.pkl"
REQUIRED_DATASET_COLUMNS = {
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_uploaded_dataset(uploaded_file):
    return pd.read_csv(uploaded_file)


model = load_model()

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

st.title("Fraud Detection Dashboard")
st.caption(
    "Use the form for fraud prediction. Upload the dataset CSV during runtime "
    "to unlock charts without storing the large file in GitHub."
)

st.sidebar.header("Enter Transaction Details")

transaction_type = st.sidebar.selectbox(
    "Transaction Type",
    ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"],
)

amount = st.sidebar.number_input("Amount", min_value=0.0)
oldbalanceOrg = st.sidebar.number_input("Old Balance Origin", min_value=0.0)
newbalanceOrig = st.sidebar.number_input("New Balance Origin", min_value=0.0)
oldbalanceDest = st.sidebar.number_input("Old Balance Destination", min_value=0.0)
newbalanceDest = st.sidebar.number_input("New Balance Destination", min_value=0.0)

input_data = pd.DataFrame(
    {
        "type": [transaction_type],
        "amount": [amount],
        "oldbalanceOrg": [oldbalanceOrg],
        "newbalanceOrig": [newbalanceOrig],
        "oldbalanceDest": [oldbalanceDest],
        "newbalanceDest": [newbalanceDest],
    }
)

if st.sidebar.button("Predict Fraud"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"Fraud detected. Probability: {probability:.2f}")
    else:
        st.success(f"Safe transaction. Fraud probability: {probability:.2f}")

st.header("Data Insights")
uploaded_dataset = st.file_uploader(
    "Upload dataset CSV",
    type=["csv"],
    help="Upload the original dataset here after deployment or when running locally.",
)

if uploaded_dataset is None:
    st.info(
        "No dataset uploaded yet. The app can still predict fraud, but the charts "
        "will appear only after you upload the CSV file."
    )
else:
    df = load_uploaded_dataset(uploaded_dataset)
    missing_columns = REQUIRED_DATASET_COLUMNS.difference(df.columns)

    if missing_columns:
        st.error(
            "Uploaded CSV is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Transaction Types")
            fig1, ax1 = plt.subplots()
            df["type"].value_counts().plot(kind="bar", ax=ax1)
            st.pyplot(fig1)
            plt.close(fig1)

        with col2:
            st.subheader("Fraud Distribution")
            fig2, ax2 = plt.subplots()
            sns.countplot(data=df, x="isFraud", ax=ax2)
            st.pyplot(fig2)
            plt.close(fig2)

        st.subheader("Amount Distribution (Log Scale)")
        fig3, ax3 = plt.subplots()
        sns.histplot(np.log1p(df["amount"]), bins=100, kde=True, ax=ax3)
        st.pyplot(fig3)
        plt.close(fig3)

        st.subheader("Correlation Heatmap")
        corr = df[
            [
                "amount",
                "oldbalanceOrg",
                "newbalanceOrig",
                "oldbalanceDest",
                "newbalanceDest",
                "isFraud",
            ]
        ].corr()

        fig4, ax4 = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax4)
        st.pyplot(fig4)
        plt.close(fig4)
