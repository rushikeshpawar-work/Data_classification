import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV


# 🔧 Preprocessing Function (no manual encoding)
def preprocess_data(df, label_encoder=None, is_train=True):
    df = df.copy()

    # --- Handle Dates ---
    df['Date of Admission'] = pd.to_datetime(df['Date of Admission'], errors='coerce')
    df['Discharge Date'] = pd.to_datetime(df['Discharge Date'], errors='coerce')
    df['Length of Stay'] = (df['Discharge Date'] - df['Date of Admission']).dt.days

    # --- Drop Irrelevant Columns ---
    drop_cols = [
        'Name', 'Blood Type', 'Date of Admission', 'Doctor', 'Hospital',
        'Insurance Provider', 'Billing Amount', 'Room Number', 'Discharge Date'
    ]
    df.drop(columns=drop_cols, inplace=True)

    # --- Encode Target ---
    if is_train:
        label_encoder = LabelEncoder()
        df['test_results'] = label_encoder.fit_transform(df['Test Results'])
    else:
        df['test_results'] = label_encoder.transform(df['Test Results'])

    # --- Split features and target ---
    X = df.drop(columns=['Test Results', 'test_results'])
    y = df['test_results']
    return X, y, label_encoder


# Get categorical feature names (for CatBoost)
def get_cat_features(X):
    return X.select_dtypes(include='object').columns.tolist()


# 🏋️‍♂️ Training Function
def train_model(X, y, cat_features):
    model = CatBoostClassifier(
        iterations=900,
        learning_rate=0.4,
        depth=9,
        l2_leaf_reg=9,
        loss_function='MultiClass',
        eval_metric='MultiClass',
        boosting_type= 'Plain',
        random_seed=42,
        verbose=100
    )

    train_pool = Pool(X, y, cat_features=cat_features)
    model.fit(train_pool)

    return model


#  Save model and label encoder
def save_artifacts(model, label_encoder, model_path='model.cbm', encoder_path='label_encoder.pkl'):
    model.save_model(model_path)
    joblib.dump(label_encoder, encoder_path)


#  Load model and label encoder
def load_artifacts(model_path='model.cbm', encoder_path='label_encoder.pkl'):
    model = CatBoostClassifier()
    model.load_model(model_path)
    label_encoder = joblib.load(encoder_path)
    return model, label_encoder


#  Predict and Append Results
def predict_and_append(df, model, label_encoder):
    X, _, _ = preprocess_data(df, label_encoder=label_encoder, is_train=False)
    cat_features = get_cat_features(X)

    preds = model.predict(X).flatten()
    probs = model.predict_proba(X).max(axis=1)
    pred_labels = label_encoder.inverse_transform(preds.astype(int))

    output_df = df.copy()
    output_df['Predicted Test Result'] = pred_labels
    output_df['Confidence Level'] = probs

    return output_df


def evaluate_model(model, X, y_true, label_encoder=None):
    """
    Evaluate model accuracy and print classification report.
    Optionally decode labels with LabelEncoder.
    """
    # Predict class indices
    y_pred = model.predict(X).astype(int).flatten()

    # Decode labels if label encoder is provided
    if label_encoder:
        y_true = label_encoder.inverse_transform(y_true)
        y_pred = label_encoder.inverse_transform(y_pred)

    # Accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"\n Accuracy: {acc:.4f}\n")

    # Classification report
    print(" Classification Report:")
    print(classification_report(y_true, y_pred))

    return acc

#Hyperparameter tunning
def tune_model(X, y,cat_features):
    model = CatBoostClassifier(
        loss_function='MultiClass',
        eval_metric='MultiClass',
        verbose=100,
        random_seed=42,
        cat_features=cat_features
    )

    param_grid = {
        'iterations': [800,900,1000],
        'learning_rate': [0.2,0.3,0.4,0.5],
        'depth': [7,8,9],
        'l2_leaf_reg': [5,7,9],
        'boosting_type': ['Ordered', 'Plain']
    }

    # grid = GridSearchCV(
    #     estimator=model,
    #     param_grid=param_grid,
    #     cv=3,
    #     scoring='accuracy',
    #     n_jobs=-1
    # )

    # grid.fit(X, y)

    from sklearn.model_selection import StratifiedKFold

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=108,  # Try 600 random combinations
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2,
        random_state=42
    )

    random_search.fit(X, y)


    # print("\n✅ Best Parameters:", grid.best_params_)
    # print("✅ Best CV Accuracy:", grid.best_score_)

    print("Best parameters:", random_search.best_params_)
    print("Best accuracy:", random_search.best_score_)

    return random_search.best_estimator_

#  MAIN EXECUTION
if __name__ == "__main__":
    # Load and preprocess
    df = pd.read_csv("D:\\ML Poc\\archive\\healthcare_dataset.csv")
    X, y, label_encoder = preprocess_data(df, is_train=True)
    cat_features = get_cat_features(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train
    model = train_model(X_train, y_train, cat_features)

    # Save artifacts
    save_artifacts(model, label_encoder)

    # Reload (to simulate real deployment)
    loaded_model, loaded_le = load_artifacts()

    # Predict and add results to DataFrame
    final_df = predict_and_append(df, loaded_model, loaded_le)
    # print(final_df.head())

    # Optional: save to file
    # final_df.to_csv("predicted_results.csv", index=False)

    evaluate_model(model, X_test, y_test, label_encoder=None)

    # print(tune_model(X_train, y_train,cat_features))
