# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 13:50:29 2024

@author: ALFIYA
"""

import numpy as np
import pickle
import streamlit as st


# loading the saved model
loaded_model = pickle.load(open('C:/Users/ALFIYA/trained_modell.sav', 'rb'))


# creating a function for prediction

def wine_quality(input_data):
    
    # loading the saved model
    loaded_model = pickle.load(open('C:/Users/ALFIYA/trained_modell.sav', 'rb'))


    input_data = (7.1,0.750,0.01,2.2,0.059,7)
    
    
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0]==1):
      return 'Good Quality Wine'
    else:
      return 'Bad Quality Wine'
      
      
      
def main():
    
    
    #giving a title
    st.title('Wine Quality Prediction Web App')
    
    
    #getting the input data from the data
    
    fixed_acidity = st.text_input('Value of fixed acidity',key = 'fixed_acidity')
    citric_acid = st.text_input('Value of citric acid',key = 'citric_acid')
    residual_sugar = st.text_input('Value of residual sugar',key = 'residual_sugar')
    sulphates = st.text_input('Value of sulphates',key = 'sulphates')
    alcohol = st.text_input('Value of alcohol',key = 'alcohol')
    quality = st.text_input('Value of quality',key = 'quality')
    
    
    
    
    # creating a button for prediction
    if st.button('Quality of Wine'):
        input_data = [fixed_acidity, citric_acid,residual_sugar, sulphates, alcohol, quality]
        diagnosis = wine_quality(input_data)
        st.success(f'Wine Quality: {diagnosis}')
        
    
    



if __name__ == '__main__':
    main()


      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      