# Brain_Stroke_Prediction
## Overview

This project focuses on predicting the likelihood of a brain stroke based on various health parameters. It utilizes machine learning algorithms to analyze patient data and make predictions, aiming to assist healthcare professionals in early diagnosis and intervention.

## Dataset

The dataset used in this project contains various features related to stroke risk factors and symptoms, including:

- **Hypertension**
- **Age**
- **Ever Married**
- **Average Glucose Level**
- **Smoking Status**
- **BMI**
- **Lifestyle**
- **Stroke History**
- **Family Stroke History**

The dataset consists of 43,400 rows and 12 columns before preprocessing.

- [Dataset File: Brain_stroke_dataset.csv](https://github.com/AartiM03/Brain_Stroke_Prediction/blob/main/Brain_stroke_dataset.csv)

## Models Used

Some machine learning models that were trained and evaluated are:

- Decision Tree
- Random Forest
- Logistic Regression
- K-Nearest Neighbors (KNN)
- XGBoost (Best Performing Model)

## Evaluation Metrics

The models were evaluated based on the following metrics:

- Accuracy
- Precision
- Recall
- F1 Score

Additionally, confusion matrices and ROC curves were plotted to visualize the performance.

## Data Preprocessing

To ensure the data quality and enhance model performance, the following preprocessing steps were performed:

- Handling missing values
- Dropping 'id' column
- Addressing outliers in 'bmi' and 'avg_glucose_level' columns
- Balancing the dataset
- Scaling the features

## Feature Selection

Various feature selection methods were applied to determine the most significant features, including:

- Information Gain
- Chi-Square
- ANOVA

## Cross-Validation

K-fold cross-validation was used to validate the models and ensure their robustness.

## User Interface

A user-friendly interface was created using Streamlit, allowing users to input their health parameters and receive stroke risk predictions. The interface demonstrates the model's accuracy and practical application by using values from the training and test datasets.
