import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="Fraud Detection", layout="wide")

st.title("💳 Real-Time Fraud Detection System")

API_URL = "http://127.0.0.1:9000"

# -------------------------
# Upload CSV
# -------------------------
uploaded_file = st.file_uploader("Upload Transaction CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    if st.button("Predict Fraud for All Rows"):

        with st.spinner("Processing..."):

            try:
                # 🚀 BATCH CALL (ONLY CHANGE HERE)
                response = requests.post(
                    f"{API_URL}/predict_batch",
                json={"features": df.values.tolist()},
                    timeout=300
                )

                if response.status_code == 200:
                    results = response.json()
                    result_df = pd.DataFrame(results)

                    final_df = pd.concat(
                        [df.reset_index(drop=True), result_df],
                        axis=1
                    )

                    st.dataframe(final_df)

                else:
                    st.error("Batch prediction failed.")

            except requests.exceptions.RequestException:
                st.error("Backend connection failed.")

# -------------------------
# Manual Entry
# -------------------------
st.markdown("---")
st.subheader("Manual Transaction Entry")

manual_input = st.text_area(
    "Enter comma-separated feature values",
    height=120,
)

if st.button("Predict Manual Transaction"):

    try:
        import re

        raw_values = re.split(r"[,\t]+", manual_input.strip())

        features = []
        for val in raw_values:
            val = val.replace(" ", "")
            if val != "":
                features.append(float(val))

        if len(features) != 30:
            st.warning("⚠ Please enter exactly 30 feature values.")
        else:

            with st.spinner("Predicting..."):

                response = requests.post(
                    f"{API_URL}/predict",
                json={"features": features},
                    timeout=5
                )

                if response.status_code == 200:

                    result = response.json()

                    if result["prediction"] == 1:
                        st.error("🚨 FRAUD DETECTED!")
                    else:
                        st.success("✅ Legitimate Transaction")

                    st.write(
                        "Fraud Probability:",
                        round(result["fraud_probability"], 4)
                    )

                else:
                    st.error("Backend error.")

    except ValueError:
        st.warning("Invalid input format. Enter numbers only.")
    except requests.exceptions.RequestException:
        st.error("Could not connect to backend.")
