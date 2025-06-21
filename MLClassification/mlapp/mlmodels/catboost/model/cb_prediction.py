import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostClassifier



# 🔧 Preprocessing Function (no manual encoding)
def preprocess_data(df):
    df = df.copy()

    # --- Handle Dates ---
    df['Date of Admission'] = pd.to_datetime(df['Date of Admission'], errors='coerce')
    df['Discharge Date'] = pd.to_datetime(df['Discharge Date'], errors='coerce')
    df['Length of Stay'] = (df['Discharge Date'] - df['Date of Admission']).dt.days

    # --- Drop Irrelevant Columns ---
    drop_cols = [
        'Name', 'Blood Type', 'Date of Admission', 'Doctor', 'Hospital',
        'Insurance Provider', 'Billing Amount', 'Room Number', 'Discharge Date']
    
    df.drop(columns=drop_cols, inplace=True)

    return df


#  Load model and label encoder
def load_artifacts(model_path='D:\ML Poc\Data_classification\MLClassification\mlapp\mlmodels\catboost\data\model.cbm', encoder_path='D:\ML Poc\Data_classification\MLClassification\mlapp\mlmodels\catboost\data\label_encoder.pkl'):
    model = CatBoostClassifier()
    model.load_model(model_path)
    label_encoder = joblib.load(encoder_path)
    return model, label_encoder


#  Predict and Append Results
def predict_and_append(df):

    model, label_encoder = load_artifacts()

    X = preprocess_data(df)
    
    preds = model.predict(X).flatten()
    probs = model.predict_proba(X).max(axis=1)
    pred_labels = label_encoder.inverse_transform(preds.astype(int))

    output_df = df.copy()
    output_df['Predicted Class'] = pred_labels
    output_df['Confidence(%)'] = np.round(probs*100)

    average_confidence = output_df['Confidence(%)'].mean()

    return output_df,np.round(average_confidence)




  

   
