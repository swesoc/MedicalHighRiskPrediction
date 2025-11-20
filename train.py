from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import pandas as pd
from sklearn.model_selection import train_test_split



df = pd.read_csv('data/training_data.csv')
df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=1)
df_train = df_train.reset_index(drop=True)

y_train = df_train['is_high_risk'].values
y_val = df_val['is_high_risk'].values

df_train = df_train.drop(["risk_score", "is_high_risk"], axis=1)
df_val = df_val.drop(["risk_score", "is_high_risk"], axis=1)

categorical_cols = df_train.select_dtypes(include=["object", "category"]).columns
numerical_cols = df_train.select_dtypes(include=["int64", "float64"]).columns

# ---------------------------------------------------
# 1. Modular Preprocessor
# ---------------------------------------------------
def build_preprocessor(categorical_cols, numerical_cols):
    return ColumnTransformer([
        ('encoder', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('scaler', StandardScaler(), numerical_cols)
    ])


# ---------------------------------------------------
# 2. Modular Model Builder
#    model_type options:
#    - 'lr'          → plain Logistic Regression
#    - 'lr_l1'       → L1 regularization
#    - 'lr_l2'       → L2 regularization
# ---------------------------------------------------
def build_model(model_type='lr'):
    if model_type == 'lr':
        return LogisticRegression(max_iter=1000)

    elif model_type == 'lr_l1':
        return LogisticRegression(
            penalty='l1',
            solver='liblinear',
            max_iter=1000
        )

    elif model_type == 'lr_l2':
        return LogisticRegression(
            penalty='l2',
            solver='lbfgs',
            max_iter=1000
        )

    # Decision tree classifier
    elif model_type == 'dt':
        return DecisionTreeClassifier(random_state=42)

    # Random forest classifier
    elif model_type == 'rf':
        return RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )


    else:
        raise ValueError("Unknown model type")


# ---------------------------------------------------
# 3. Build full pipeline
# ---------------------------------------------------
def build_pipeline(preprocessor, model):
    return Pipeline([
        ('preprocess', preprocessor),
        ('clf', model)
    ])


# Preprocessing (shared across both models)
preprocessor = build_preprocessor(categorical_cols, numerical_cols)

# ---------------------------------------------------
# PIPELINE 1 — Logistic Regression (no regularization)
# ---------------------------------------------------
model_lr = build_model('lr')
pipeline_lr = build_pipeline(preprocessor, model_lr)

pipeline_lr.fit(df_train, y_train)

# ---------------------------------------------------
# Performance Evaluation
# ---------------------------------------------------
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)

    return {
        #'Accuracy': accuracy_score(y_val, y_pred),
        'Precision': precision_score(y_val, y_pred, average='binary'),
        'Recall': recall_score(y_val, y_pred, average='binary'),
        'F1 Score': f1_score(y_val, y_pred, average='binary')
    }

# =============================
# Create output directory
# =============================
output_dir = "output_models"
os.makedirs(output_dir, exist_ok=True)

# List of models to compare
model_types = ['lr', 'lr_l1', 'dt', 'rf']
models = {
    "logistic_regression": 'lr',
    "logistic_regression with regularization": 'lr_l1',
    "decision_tree": 'dt',
    "random_forest": 'rf'

}

results = {}

for name, m in models.items():
    model = build_model(m)
    pipe = build_pipeline(preprocessor, model)
    pipe.fit(df_train, y_train)

    # store model performance
    results[m] = evaluate_model(pipe, df_val, y_val)

    # Construct file path
    file_path = os.path.join(output_dir, f"{name}.pkl")

    # Save the model
    joblib.dump(model, file_path)
    print(f"Saved: {file_path}")


# Convert results to DataFrame
results_df = pd.DataFrame(results).T
print(results_df)




print(f'\nModel saved to {output_dir}!')


