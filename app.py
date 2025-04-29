import streamlit as st
import pickle
import numpy as np

# Load the trained model
model_filename = "trained_model.sav"
loaded_model = pickle.load(open(model_filename, 'rb'))

# Define a function for prediction
def wine_quality_prediction(input_data):
    input_data = np.array(input_data).reshape(1, -1)
    prediction = loaded_model.predict(input_data)
    return "Good Quality Wine" if prediction[0] == 1 else "Bad Quality Wine"

# Streamlit app
st.title("Red Wine Quality Prediction")

st.write("""
### Enter the features of the wine to predict its quality:
""")

# Input fields for user data
fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, step=0.1, format="%.2f")
volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, step=0.1, format="%.2f")
citric_acid = st.number_input("Citric Acid", min_value=0.0, step=0.1, format="%.2f")
residual_sugar = st.number_input("Residual Sugar", min_value=0.0, step=0.1, format="%.2f")
chlorides = st.number_input("Chlorides", min_value=0.0, step=0.001, format="%.3f")
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0.0, step=1.0, format="%.1f")
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0, step=1.0, format="%.1f")
density = st.number_input("Density", min_value=0.0, step=0.0001, format="%.4f")
pH = st.number_input("pH", min_value=0.0, step=0.01, format="%.2f")
sulphates = st.number_input("Sulphates", min_value=0.0, step=0.01, format="%.2f")
alcohol = st.number_input("Alcohol", min_value=0.0, step=0.1, format="%.2f")

# Prediction button
if st.button("Predict Wine Quality"):
    # Collect input data
    input_features = [
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
        free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol
    ]
    
    # Check for missing values
    if any(f is None for f in input_features):
        st.warning("Please fill in all the fields to make a prediction.")
    else:
        # Make prediction
        prediction = wine_quality_prediction(input_features)
        st.success(f"The predicted wine quality is: {prediction}")
