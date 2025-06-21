import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def preprocess(df,expected_columns,scaler):

    # Handle date columns
    df['Date of Admission'] = pd.to_datetime(df['Date of Admission'], errors='coerce')
    df['Discharge Date'] = pd.to_datetime(df['Discharge Date'], errors='coerce')
    df['Length of Stay'] = (df['Discharge Date'] - df['Date of Admission']).dt.days

    # Drop non-relevant or high-cardinality columns
    drop_columns = [
        'Name', 'Blood Type', 'Date of Admission', 'Doctor', 'Hospital',
        'Insurance Provider', 'Billing Amount', 'Room Number', 'Discharge Date'
    ]
    df = df.drop(columns=drop_columns)

    # One-hot encode categorical columns
    categorical_cols = ['Gender', 'Medical Condition', 'Admission Type', 'Medication']
    df = pd.get_dummies(df, columns=categorical_cols)

    # Scale numeric columns
    df['Age'] = scaler.fit_transform(df[['Age']])
    df['Length of Stay'] = scaler.fit_transform(df[['Length of Stay']])

    #adding missing columns which are made during one-hot encoding but not received in data
    df = df.reindex(columns=expected_columns, fill_value=0)

    return df


def predict_and_annotate_df(model_path, inputdf):
    """
    Loads a model and label encoder from a joblib file, makes predictions,
    and appends predicted class and confidence to the original DataFrame.

    Parameters:
    - model_path: Path to .joblib file containing {'model': ..., 'label_encoder': ...}
    - X: Features for prediction (array-like or DataFrame)
    - original_df: Original DataFrame to which results will be added.
    - top_n: (optional) If set, prints top N most confident predictions for inspection.

    Returns:
    - DataFrame: original_df + 'Predicted Class' + 'Confidence'
    """

     # Load model and label encoder
    saved = joblib.load(model_path)
    model = saved['model']
    label_encoder = saved.get('label_encoder', None)
    expected_columns = saved.get('columns')
    scaler = saved.get('scaler') 

    #process the input data
    process_df = preprocess(inputdf,expected_columns,scaler)
    
   
    # Predict
    probas = model.predict_proba(process_df)
    predictions = np.argmax(probas, axis=1)
    confidences = np.max(probas, axis=1)

    # Decode labels 
    predicted_labels = label_encoder.inverse_transform(predictions)

    # Return merged DataFrame
    result_df = inputdf.copy().reset_index(drop=True)
    result_df['Predicted Class'] = predicted_labels
    result_df['Confidence(%)'] = np.round(confidences*100)

    average_confidence = result_df['Confidence(%)'].mean()

    return result_df,np.round(average_confidence)


