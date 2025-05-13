import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# ----------------------
# Load and preprocess data
# ----------------------
def preprocess(df,label_encoder=None):

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

    # Encode target variable
    if label_encoder is None:
        label_encoder = LabelEncoder()
        df['test_results'] = label_encoder.fit_transform(df['Test Results'])
    else:
        df['test_results'] = label_encoder.transform(df['Test Results'])


    # Scale numeric columns
    scaler = StandardScaler()
    df['Age'] = scaler.fit_transform(df[['Age']])
    df['Length of Stay'] = scaler.fit_transform(df[['Length of Stay']])

    return df, label_encoder,scaler


# ----------------------
# Train/Test Split
# ----------------------
def split_data(filepath):
    raw_df = pd.read_csv(filepath)

    # Split raw DataFrame before preprocessing
    df_train_raw, df_test_raw = train_test_split(
        raw_df, test_size=0.2, random_state=42, stratify=raw_df['Test Results'])
    print(df_test_raw)
    return df_train_raw, df_test_raw


# ----------------------
# Model Training and Evaluation
# ----------------------
def train_and_evaluate_rf(X_train, X_test, y_train, y_test, label_encoder,scaler):
    model = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        min_samples_split=2,
        max_depth=35,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # save model in joblib file
    joblib.dump({
        'model': model,
        'label_encoder': label_encoder,
        'scaler': scaler,
        'columns': X_train.columns.tolist()
    }, 'RF_model.joblib')

    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    return model


# ----------------------
# Hyperparameter Tuning
# ----------------------
def tune_hyperparameters(X_train, y_train):
    param_grid = {
        'n_estimators': [500],
        'max_depth': [10, 15, 25, 30, 35, 50, None],
        'min_samples_leaf': [2],
        'min_samples_split': [2, 5],
        'max_features': ['log2', 'sqrt']
    }
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=3,
        scoring='f1_macro',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    print("Best Parameters:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)


# ----------------------
# calculate model confidence
# ----------------------

def calculate_confidence(model, X, y_true=None, label_encoder=None, top_n=5):
    """
    Calculate model prediction confidence scores.

    Parameters:
    - model: Trained classifier with predict_proba method.
    - X: Features to predict on.
    - y_true: (optional) True labels for comparison/reporting.
    - label_encoder: (optional) To decode numerical labels back to original class names.
    - top_n: (optional) Number of top confident predictions to return (for inspection).

    Returns:
    - confidences: Array of max probabilities per prediction.
    - predictions: Predicted class indices.
    """
    # Get predicted probabilities
    probas = model.predict_proba(X)

    # Get the predicted class and its associated probability
    confidences = np.max(probas, axis=1)
    predictions = np.argmax(probas, axis=1)

    # Optional reporting
    if y_true is not None and label_encoder is not None:
        decoded_preds = label_encoder.inverse_transform(predictions)
        decoded_true = label_encoder.inverse_transform(y_true)
        for i in np.argsort(confidences)[-top_n:][::-1]:  # Top N most confident
            print(f"Prediction: {decoded_preds[i]}, True: {decoded_true[i]}, Confidence: {confidences[i]:.4f}")

    return confidences, predictions

# ----------------------
# calculate model prediction_summary
# ----------------------

def prediction_summary_df(model, X, original_df, label_encoder=None):
    """
    Returns a DataFrame with predicted class and confidence for each row in X.

    Parameters:
    - model: Trained classifier supporting predict_proba().
    - X: Feature matrix used for prediction (should align with original_df).
    - original_df: DataFrame before dropping columns (or matching index).
    - label_encoder: (Optional) If labels are encoded, use to decode predictions.

    Returns:
    - result_df: Original DataFrame with added 'Predicted Class' and 'Confidence' columns.
    """
    # Predict probabilities
    probas = model.predict_proba(X)
    predictions = np.argmax(probas, axis=1)
    confidences = np.max(probas, axis=1)

    # Decode if label encoder is available
    if label_encoder:
        predicted_labels = label_encoder.inverse_transform(predictions)
    else:
        predicted_labels = predictions

    # Create copy of original DataFrame to avoid modifying input
    result_df = original_df.copy().reset_index(drop=True)


    # Append predictions and confidence
    result_df['Predicted Class'] = predicted_labels
    result_df['Confidence'] = confidences

    return result_df


# ----------------------
# Main Execution
# ----------------------
def main():
    file_path = "D:\\ML Poc\\archive\\healthcare_dataset.csv"
    # df, label_encoder = preprocess(file_path)

    df_train_raw, df_test_raw = split_data(file_path)

    df_test_raw_copy = df_test_raw.drop(columns=['Test Results'])

    # preprocess
    train_process_df,label_encoder,scaler = preprocess(df_train_raw,label_encoder=None)
    test_process_df,label_encoder,scaler = preprocess(df_test_raw,label_encoder=label_encoder)

    # X & Y split

    X_train = train_process_df.drop(columns=['Test Results', 'test_results'])
    y_train = train_process_df['test_results']

    X_test = test_process_df.drop(columns=['Test Results', 'test_results'])
    y_test = test_process_df['test_results']

    rf_model = train_and_evaluate_rf(X_train, X_test, y_train, y_test, label_encoder,scaler)

    # tune_hyperparameters(X_train, y_train)
    
    confidences, predictions = calculate_confidence(rf_model, X_test, y_test, label_encoder=label_encoder, top_n=5)
    print("Average model confidence:", np.mean(confidences))
    

    result_df = prediction_summary_df(rf_model, X_test, original_df=df_test_raw_copy, label_encoder=label_encoder)

    print(result_df)

if __name__ == "__main__":
    main()
