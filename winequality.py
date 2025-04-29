# -*- coding: utf-8 -*-
"""
Red Wine Quality Prediction Script

This script predicts wine quality using machine learning models.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score
import pickle

# Ensure the working directory is set correctly
current_dir = os.path.dirname(__file__)
data_path = os.path.join(current_dir, "data", "winequality-red.csv")

# Load the dataset
df = pd.read_csv(data_path)

# Data Preprocessing
df = df.rename(columns={
    'fixed acidity': 'fixed_acidity',
    'volatile acidity': 'volatile_acidity',
    'citric acid': 'citric_acid',
    'residual sugar': 'residual_sugar',
    'free sulfur dioxide': 'free_sulfur_dioxide',
    'total sulfur dioxide': 'total_sulfur_dioxide'
})

# Add a "good_quality" column
df['good_quality'] = [1 if x >= 7 else 0 for x in df['quality']]

# Feature-target split
X = df.drop(['good_quality', 'quality'], axis=1)
y = df['good_quality']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Train a Random Forest Classifier
rf = RandomForestClassifier(random_state=42)
rf.fit(x_train, y_train)

# Model Evaluation
train_pred = rf.predict(x_train)
test_pred = rf.predict(x_test)

print("Model Accuracy on Train Set:", accuracy_score(y_train, train_pred))
print("Model Accuracy on Test Set:", accuracy_score(y_test, test_pred))
print("Kappa Score on Test Set:", cohen_kappa_score(y_test, test_pred))
print(classification_report(y_test, test_pred))

# Save the trained model
model_filename = os.path.join(current_dir, "trained_model.sav")
with open(model_filename, 'wb') as model_file:
    pickle.dump(rf, model_file)

print(f"Model saved to {model_filename}")

# Predict function
def wine_quality(input_data):
    """
    Predict the quality of wine based on input data.
    
    Args:
        input_data (list/tuple/dict): Wine features as a list, tuple, or dictionary.
        
    Returns:
        str: Quality of the wine ("Good Quality Wine" or "Bad Quality Wine").
    """
    # Ensure all inputs are numeric and prepare data
    if isinstance(input_data, dict):
        feature_names = [
            'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
            'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
            'pH', 'sulphates', 'alcohol'
        ]
        input_data = [input_data.get(feature, 0) for feature in feature_names]
    elif isinstance(input_data, (list, tuple)):
        input_data = [float(i) for i in input_data]
    else:
        raise ValueError("Input data must be a dictionary, list, or tuple.")
    
    input_data = np.array(input_data).reshape(1, -1)
    
    # Load the model
    loaded_model = pickle.load(open(model_filename, 'rb'))
    
    # Make the prediction
    prediction = loaded_model.predict(input_data)
    return "Good Quality Wine" if prediction[0] == 1 else "Bad Quality Wine"

# Test the prediction function
test_input = {
    'fixed_acidity': 7.4,
    'volatile_acidity': 0.7,
    'citric_acid': 0.0,
    'residual_sugar': 1.9,
    'chlorides': 0.076,
    'free_sulfur_dioxide': 11.0,
    'total_sulfur_dioxide': 34.0,
    'density': 0.9978,
    'pH': 3.51,
    'sulphates': 0.56,
    'alcohol': 9.4
}

result = wine_quality(test_input)
print(f"Prediction for input {test_input}: {result}")
