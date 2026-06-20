"""
Model Training Script

This script trains and tunes three classification models to predict
next-day stock price direction (up or down).

Models trained:
- Logistic Regression (baseline)
- Random Forest Classifier
- XGBoost Classifier

Each model is tuned using GridSearchCV with TimeSeriesSplit to prevent
data leakage. Trained models are saved to the models/ folder.
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier

# Load processed dataset
df = pd.read_csv("data/processed/processedData.csv")

# Split data chronologically to prevent data leakage
train_df = df[df["Year"] < 2021]
val_df = df[(df["Year"] >= 2021) & (df["Year"] < 2023)]
test_df = df[df["Year"] >= 2023]

# Drop Year column as it is not a predictive feature
train_df = train_df.drop(columns='Year')
val_df = val_df.drop(columns='Year')
test_df = test_df.drop(columns='Year')

# Separate features and target variable
train_inputs = train_df.drop(columns=['Target'])
train_target = train_df['Target']

val_inputs = val_df.drop(columns=['Target'])
val_target = val_df['Target']

test_inputs = test_df.drop(columns=['Target'])
test_target = test_df['Target']

# Fit scaler on training data only to prevent data leakage
scaler = StandardScaler()
scaler.fit(train_inputs)

# Apply same scaling to all splits
train_inputs = scaler.transform(train_inputs)
val_inputs = scaler.transform(val_inputs)
test_inputs = scaler.transform(test_inputs)

# --- Logistic Regression ---
base_LR = LogisticRegression(random_state=42)

param_grid_LR = {
    'C': [0.01, 0.1, 1, 10],
    'max_iter': [100, 200, 500]
}

grid_LR = GridSearchCV(estimator=base_LR, param_grid=param_grid_LR, cv=TimeSeriesSplit(n_splits=5), scoring='accuracy', n_jobs=-1)
grid_LR.fit(train_inputs, train_target)
model_LR = grid_LR.best_estimator_
print("The Best Params for Logistic Regression are: \n", grid_LR.best_params_)

# --- Random Forest ---
base_RFC = RandomForestClassifier(random_state=42)

param_grid_RFC = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10]
}

grid_RFC = GridSearchCV(estimator=base_RFC, param_grid=param_grid_RFC, cv=TimeSeriesSplit(n_splits=5), scoring='accuracy', n_jobs=-1)
grid_RFC.fit(train_inputs, train_target)
model_RFC = grid_RFC.best_estimator_
print("The Best Params for Random Forest are: \n", grid_RFC.best_params_)

# --- XGBoost ---
base_XGB = XGBClassifier(random_state=42)

param_grid_XGB = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1],
}

grid_XGB = GridSearchCV(estimator=base_XGB, param_grid=param_grid_XGB, cv=TimeSeriesSplit(n_splits=5), scoring='accuracy', n_jobs=-1)
grid_XGB.fit(train_inputs, train_target)
model_XGB = grid_XGB.best_estimator_
print("The Best Params for XGBoost are: \n", grid_XGB.best_params_)

# Save trained models to disk
joblib.dump(model_LR, 'models/model_LR.joblib')
joblib.dump(model_RFC, 'models/model_RFC.joblib')
joblib.dump(model_XGB, 'models/model_XGB.joblib')