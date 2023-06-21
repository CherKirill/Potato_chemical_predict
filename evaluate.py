import pickle
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import json


def predict(model_path: str,
            X_path: str,
            X_scaler_path: str,
            y_scaler_path: str,
            save_pred_to: Optional[str] = None):

    # Load objects
    model = joblib.load(model_path)
    X = pd.read_csv(X_path, index_col=0)  # if any error, try removing index_col=0 from brackets
    with open(X_scaler_path, 'rb') as X_scaler_bytes:
        X_scaler = pickle.load(X_scaler_bytes)
    with open(y_scaler_path, 'rb') as y_scaler_bytes:
        y_scaler = pickle.load(y_scaler_bytes)

    # Transform numerical columns of training data
    numerical_cols = ['clr_N', 'clr_P', 'clr_K', 'clr_Ca', 'clr_Mg', 'clr_Fv']
    X[numerical_cols] = X_scaler.fit_transform(X[numerical_cols])

    # Get predictions and transform them back to original scale
    pred = model.predict(X)
    pred = y_scaler.inverse_transform([pred])

    # Save predictions if needed
    if save_pred_to is not None:
        np.save(save_pred_to, pred)
        return
    return pred

"""
    Function for predicting y
    Args:
        model_path: Path to trained sklearn model for predicting y.
        X_path: Path to .csv of data to predict from.
        X_scaler_path: Path to scaler that needs to be applied to X.
        y_scaler_path: Path to scaler that needs to be applied to y.
        save_pred_to: Save or not to save the prediction.

    Returns:
        Saves .npy file if save_pred=True else returns numpy array with predictions.
"""

def pred(model_path,X_path,X_scaler_path,y_scaler_path):
    predict(model_path, X_path, X_scaler_path, y_scaler_path, save_pred_to="pred.npy")
    pred = np.load("pred.npy")
    data = pred[0]

    m=[]
    for i in range(0,len(data)):
        m.append(str(data[i]))
    return(m)

def out(m,save_path):
    with open(save_path, 'w') as fp:
        for line in m:
            fp.write(line + '\n')
    
if __name__ == "__main__":
    name_function = ["Vendable","Petit","Moy","Gros"]
    for i in name_function:
        mas_predict = pred("models/randomForest_Rend"+i+".pkl","X_test.csv","models/X_scaler.pickle","models/Rend"+i+"_scaler.pickle") 
        out(mas_predict,'result/Predict_Rend'+i+'.txt')
