# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 13:41:38 2024

@author: ALFIYA
"""

import numpy as np
import pickle


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
  print('Good Quality Wine')
else:
  print('Bad Quality Wine')

