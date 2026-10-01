import streamlit as st
import pandas as pd
import pickle
import os

# Load trained model
model_path = os.path.join("..", "models", "heart_model.pkl")
model = pickle.load(open(model_path, "rb"))

st.title("❤️ Heart Disease Prediction App")
st.write("Fill in the patient details below to predict heart disease risk:")

# Input fields
age = st.number_input("Age", 1, 120)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type (0-3)", [0,1,2,3])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)")
chol = st.number_input("Serum Cholesterol (mg/dl)")
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0,1])
restecg = st.selectbox("Resting ECG Results (0-2)", [0,1,2])
thalach = st.number_input("Maximum Heart Rate Achieved")
exang = st.selectbox("Exercise Induced Angina", [0,1])
oldpeak = st.number_input("ST Depression Induced by Exercise")
slope = st.selectbox("Slope of the Peak Exercise ST Segment (0-2)", [0,1,2])
ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy (0-3)", [0,1,2,3])
thal = st.selectbox("Thallium Stress Test Result (1 = normal, 2 = fixed defect, 3 = reversible defect)", [1,2,3])

# Prediction button
if st.button("Predict"):
    input_data = pd.DataFrame([[age,
                                1 if sex=="Male" else 0,
                                cp,
                                trestbps,
                                chol,
                                fbs,
                                restecg,
                                thalach,
                                exang,
                                oldpeak,
                                slope,
                                ca,
                                thal]],
                              columns=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal'])
    
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.error("⚠️ High risk of heart disease")
    else:
        st.success("✅ Low risk of heart disease")
