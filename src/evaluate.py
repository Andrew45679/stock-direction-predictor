"""
Model Evaluation Script

This script evaluates trained models on held-out test data and generates
confusion matrices and classification reports for each model.
"""
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the three different models
model_LR = joblib.load('models/model_LR.joblib')
model_RFC = joblib.load('models/model_RFC.joblib')
model_XGB = joblib.load('models/model_XGB.joblib')

# Load processed dataset
df = pd.read_csv("data/processed/processedData.csv")

# Split data chronologically
test_df = df[df["Year"] >= 2023].drop(columns='Year')

# Separate features and target variable
test_inputs = test_df.drop(columns=['Target'])
test_target = test_df['Target']

# Load saved scaler and transform test data
scaler = joblib.load('models/scaler.joblib')
test_inputs = scaler.transform(test_inputs)

# Function to print the accuracy, classification report, and to make a heatmap 
def predict_and_plot(model, inputs, targets, name=''):
    preds = model.predict(inputs)
    accuracy = accuracy_score(targets, preds)
    print("{} Accuracy: {:.2f}%".format(name, accuracy * 100))
    print(classification_report(targets, preds))
    cf = confusion_matrix(targets, preds, normalize='true')
    plt.figure()
    sns.heatmap(cf, annot=True)
    plt.xlabel('Prediction')
    plt.ylabel('Target')
    plt.title('{} Confusion Matrix'.format(name))
    plt.savefig('outputs/figures/{}_confusion_matrix.png'.format(name))
    plt.show()
    return preds

# Calling the predict_and_plot function on the three different models
predict_and_plot(model_LR, test_inputs, test_target, "Logistic Regression")
predict_and_plot(model_RFC, test_inputs, test_target, "Random Forest Classifier")
predict_and_plot(model_XGB, test_inputs, test_target, "XGBoost Classifier")