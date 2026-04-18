import streamlit as st
import numpy as np
import pickle
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("model.h5")

# Load scaler
scaler = pickle.load(open("scaler.pkl", "rb"))

# Load columns
columns = pickle.load(open("columns.pkl", "rb"))

# Title
st.title("Stroke Prediction App")
st.subheader("Enter Patient Health Details")

# User input
age = st.number_input("Age")
bmi = st.number_input("BMI")
hypertension = st.selectbox("Hypertension", [0,1])
heart_disease = st.selectbox("Heart Disease", [0,1])
avg_glucose_level = st.number_input("Glucose Level")

# Button
if st.button("Predict"):

    # Create full input (same size as training)
    input_data = np.zeros((1, len(columns)))

    # Fill values (IMPORTANT)
    input_data[0][0] = age
    input_data[0][1] = bmi

    # Scale
    input_data = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_data)

    # Result
    if prediction[0][0] > 0.1:
        st.error("Stroke Yes")
    else:
        st.success("No Stroke")

