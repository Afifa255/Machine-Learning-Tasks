# End-to-End Machine Learning Projects

This repository contains three machine learning projects covering traditional ML, multimodal deep learning, and NLP using BERT.

---

# Task 1: Customer Churn Prediction

## Objective

Develop a reusable machine learning pipeline to predict customer churn using the Telco Customer Churn dataset.

## Dataset

**Dataset:** Telco Customer Churn

The dataset contains customer demographic details, service subscriptions, and account-related information.

## Data Preprocessing

- Removed missing values and duplicate records
- Applied One-Hot Encoding
- Standardized numerical features using StandardScaler
- Built preprocessing pipeline using ColumnTransformer

## Models

- Logistic Regression
- Random Forest Classifier

## Results

| Model | Accuracy |
|--------|----------|
| Logistic Regression | 81.9% |
| Random Forest | 73.5% |

---

# Task 2: Housing Price Prediction using Images and Tabular Data

## Objective

Build a multimodal deep learning model using image and structured data.

## Features

- CNN for image processing
- Dense Network for tabular features
- Feature Fusion using Concatenate Layer

## Results

| Metric | Training | Validation |
|----------|----------|------------|
| MAE | 696K USD | 703K USD |
| RMSE | 785K USD | 796K USD |

---

# Task 3: News Topic Classification using BERT

## Objective

Fine-tune BERT on the AG News dataset to classify news headlines.

## Categories

- World
- Sports
- Business
- Sci/Tech

## Technologies

- TensorFlow
- Hugging Face Transformers
- Streamlit

## Deployment

A Streamlit application allows real-time news headline classification.
