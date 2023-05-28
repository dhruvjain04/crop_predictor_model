import pickle

import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier


st.title("Realtime crop predictor")
file = open('randomforest_model.pkl', 'rb')

loaded_model = pickle.load(open('randomforest_model.pkl', 'rb'))

N = st.number_input("Nitrogen Content of soil")
P = st.number_input("Phosphorous Content of soil")
K = st.number_input("Potassium Content of soil")
temperature = st.number_input("Current temperature of region")
humidity = st.number_input("Humidity level of region")
pH = st.number_input("pH value of soil")
rainfall = st.number_input("Amount of rainfall(in mm)")

test = [[N, P, K, temperature, humidity, pH, rainfall]]
df = pd.DataFrame(test, columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
print(df)
if st.button("Predict crop"):
    predicted_val = loaded_model.predict(df)
    for val in predicted_val:
        st.success(val)

