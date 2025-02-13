# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 13:50:29 2024

@author: ALFIYA
"""

import numpy as np
import pickle
import streamlit as st


# loading the saved model
loaded_model = pickle.load(open('C:/Users/hp/OneDrive/ドキュメント/trained_modell.sav', 'rb'))


# creating a function for prediction

def map_prediction_to_label(prediction): 
    
    if prediction == 1:
        return "Good Quality Wine"
    elif prediction == 0:
        return "Bad Quality Wine"

      
      
      
def main():
    
    
    #giving a title
    st.title('Wine Quality Prediction Web App')
    
    # User inputs for air quality indices
    st.write("Enter the values for the following air quality indices:")
    
    fixed_acidity = st.number_input('Value of fixed acidity', min_value=0.0, format="%.2f")
    citric_acid = st.number_input('Value of citric acid', min_value=0.0, format="%.2f")
    residual_sugar = st.number_input('Value of residual sugar', min_value=0.0, format="%.2f")
    sulphates = st.number_input('Value of sulphates', min_value=0.0, format="%.2f")
    alcohol = st.number_input('Value of alcohol', min_value=0.0, format="%.2f")
    quality = st.number_input('Value of quality', min_value=0.0, format="%.2f")
    
    
    
    
    # creating a button for prediction
    if st.button('Quality of Wine'):
        input_data = [fixed_acidity, citric_acid,residual_sugar, sulphates, alcohol, quality]
        
        # Predict using the loaded model
        prediction = loaded_model.predict(np.array(input_data).reshape(1, -1))[0]


        # Display the prediction
        st.success(f"The predicted Wine quality is: {map_prediction_to_label(prediction)}")
        
    
    



if __name__ == '__main__':
    main()


      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      