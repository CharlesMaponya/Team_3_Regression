"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.

    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])
    
    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------
    feature_vector_df = feature_vector_df.fillna(0, axis=1)
    feature_vector_df['train'] =0
    predict_vector = timeConstraints(feature_vector_df)
    
    
    # ----------- Replace this code with your own preprocessing steps --------
    predict_vector = predict_vector[['Destination Lat','Pickup - Day of Month','Distance (KM)','Temperature','Precipitation in millimeters','Pickup Lat','Pickup Long','Destination Long','No_Of_Orders','Age','Average_Rating','secondsUntilConfirmation','secondsUntilArrival','waitingTime']]
    # ------------------------------------------------------------------------
    
    return predict_vector

#=================================================================
# This function turns the time columns into date time objects
#=================================================================
def timeConstraints(data_param):
  """Private helper function to preprocess time columns into date time objects
    NB: 
    ----------
    data_param : Dataframe
        the data frame that contains time columns/Features
    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
       the processed features that are now converted into date time objects
    """

  ## Extracting all the time columns
  time = data_param[['Placement - Time', 'Confirmation - Time', 'Arrival at Pickup - Time', 'Pickup - Time']]

  ## Converting the Time features into Date time Features
  for col in time.columns:
      data_param[col] = pd.to_datetime(data_param[col], format='%H:%M:%S %p')
  
  # calculate duration in seconds between placement and confirmation
  data_param['daysConfirmation'] = data_param['Confirmation - Day of Month'] - data_param['Placement - Day of Month']
  data_param['secondsUntilConfirmation'] = (data_param['Confirmation - Time']-data_param['Placement - Time']).dt.seconds
  data_param['secondsUntilConfirmation'] += data_param['daysConfirmation']*86400
  del(data_param['daysConfirmation'])
  
  # calculate duration in seconds between confirmation and arrival to the pickup point
  data_param['daysArrival'] = data_param['Arrival at Pickup - Day of Month'] - data_param['Confirmation - Day of Month']
  data_param['secondsUntilArrival'] = (data_param['Arrival at Pickup - Time']-\
      data_param['Confirmation - Time']).dt.seconds
  data_param['secondsUntilArrival'] += data_param['daysArrival']*86400 
  del(data_param['daysArrival'])
  
  # calculate the waiting time for the order: time taken from the arrival at pickup until the pickup 
  data_param['waitingTime'] = (data_param['Pickup - Time']-data_param['Arrival at Pickup - Time']).dt.seconds
  
  
  return data_param

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))

def make_prediction(data, model):
    """Prepare request data for model prediciton.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standerdisation.
    return prediction[0].tolist()
