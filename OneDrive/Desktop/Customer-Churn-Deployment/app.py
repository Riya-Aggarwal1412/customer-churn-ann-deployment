import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load model
model = load_model("churn_model.h5")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Churn Predictor", layout="centered")

st.title("Bank Customer Churn Prediction")

credit_score = st.number_input("Credit Score", 300, 900)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Female", "Male"])
age = st.number_input("Age", 18, 100)
tenure = st.number_input("Tenure", 0, 10)
balance = st.number_input("Balance")
products = st.number_input("Number of Products", 1, 4)
credit_card = st.selectbox("Has Credit Card", [0,1])
active = st.selectbox("Is Active Member", [0,1])
salary = st.number_input("Estimated Salary")

gender = 1 if gender == "Male" else 0
geo_germany = 1 if geography == "Germany" else 0
geo_spain = 1 if geography == "Spain" else 0

if st.button("Predict Churn"):

    data = np.array([[credit_score, gender, age, tenure,
                      balance, products, credit_card,
                      active, salary,
                      geo_germany, geo_spain]])

    data = scaler.transform(data)

    prediction = model.predict(data)[0][0]

    churn_prob = prediction * 100
    stay_prob = (1 - prediction) * 100

    st.markdown("## Prediction")

    st.markdown(f"### Churn probability")
    st.markdown(f"# {churn_prob:.2f}%")

    st.markdown("### Stay probability")
    st.markdown(f"# {stay_prob:.2f}%")

    if prediction > 0.5:
        st.error("Decision: WILL LEAVE")
    else:
        st.success("Decision: WILL STAY")