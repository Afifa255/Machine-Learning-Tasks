Task 1: End-to-End Machine Learning Pipeline with Scikit-learn
Objective

Develop a robust and reusable machine learning pipeline to predict customer churn using the Telco Customer Churn dataset. The solution follows an end-to-end workflow, including data preprocessing, model training, hyperparameter tuning, evaluation, and model persistence.

Dataset

Dataset: Telco Customer Churn

The dataset contains customer demographic details, service subscriptions, and account-related information that can be used to predict whether a customer is likely to leave the service.

Data Preprocessing
Removed missing values and duplicate records to ensure data quality.
Applied One-Hot Encoding to transform categorical variables into numerical representations.
Standardized numerical features using StandardScaler.
Combined all preprocessing operations through a ColumnTransformer integrated within a Scikit-learn Pipeline.
Model Development

Two classification models were implemented:

Logistic Regression
Random Forest Classifier

Each model was incorporated into a complete pipeline, enabling seamless preprocessing, training, and prediction.

Hyperparameter Optimization

GridSearchCV was used to identify the optimal set of hyperparameters.

Logistic Regression
Regularization strengths (C)
Penalty types
Solver configurations
Random Forest
Number of estimators
Maximum tree depth
Feature selection strategies
Model Performance
Model	Accuracy
Logistic Regression	~81.9%
Random Forest	~73.5%

The Logistic Regression model achieved superior performance and demonstrated better generalization on unseen data.

Model Saving

The best-performing models were exported using Joblib for future deployment and inference:

logistic_regression_churn.pkl
random_forest_churn.pkl
Task 2: Multimodal Machine Learning for Housing Price Prediction
Objective

Build a multimodal deep learning model capable of predicting housing prices by combining structured tabular features with visual information extracted from house images.

Dataset
Structured Data

Housing sales dataset containing features such as:

Square Footage
Number of Bedrooms
Number of Bathrooms
City Codes
Image Data

Property images stored in the custom socal_pics/ dataset.

Each property record is linked to its corresponding image through a unique image_id.

Data Preparation
Created an image_path column to connect tabular records with image files.
Split the dataset into training (80%) and validation (20%) subsets.
Selected key numerical features:
sqft
bed
bath
n_citi
Utilized ImageDataGenerator for image preprocessing and normalization.
Feature Fusion Strategy

A custom data generator was developed to synchronize image and tabular inputs during training.

The model consists of:

Image Branch
Convolutional Neural Network (CNN)
Two convolutional layers with ReLU activation
Max Pooling layers
Flatten layer for feature extraction
Tabular Branch
Dense layers
Batch Normalization
Dropout regularization
Fusion Layer

Both feature streams are merged using a Concatenate layer before passing through fully connected layers for final price prediction.

Model Configuration
Optimizer: Adam
Loss Function: Mean Squared Error (MSE)
Metrics:
Mean Absolute Error (MAE)
Root Mean Squared Error (RMSE)
Callbacks
EarlyStopping
ReduceLROnPlateau
Training Parameters
Epochs: 15
Batch Size: 64
Image Size: 128 × 128
Results
Mean Absolute Error (MAE)
Training: ~696K USD
Validation: ~703K USD
Root Mean Squared Error (RMSE)
Training: ~785K USD
Validation: ~796K USD
Visualization

Training and validation RMSE curves were plotted across epochs. The model exhibited stable learning behavior with no significant signs of overfitting, indicating effective integration of both image and tabular modalities.

Task 3: Fine-Tuning BERT for News Topic Classification
Overview

This project demonstrates the fine-tuning of a BERT-based transformer model for automated news topic classification. The model categorizes news headlines into one of four classes:

World
Sports
Business
Science & Technology

The workflow covers data preparation, model fine-tuning, evaluation, model export, and deployment through a Streamlit web application.

Dataset

Dataset: AG News (Hugging Face Datasets)

The objective is to classify news headlines into predefined topic categories using a transformer-based architecture.

Technologies Used
TensorFlow
Hugging Face Transformers
Streamlit
Data Preprocessing
Loaded the AG News dataset using the datasets library.
Tokenized text using bert-base-uncased tokenizer.
Applied padding and truncation with a maximum sequence length of 128 tokens.
Converted processed data into TensorFlow-compatible datasets using to_tf_dataset().
Model Fine-Tuning

A TFBertForSequenceClassification model was initialized with four output classes.

Training Configuration
Optimizer: Adam
Learning Rate: 2e-5
Loss Function: SparseCategoricalCrossentropy
Metric: Accuracy
Epochs: 2
Evaluation

Model performance was assessed using:

Accuracy Score

Measures overall classification correctness.

Weighted F1-Score

Evaluates performance while accounting for class imbalance.

The evaluation results confirmed the effectiveness of the fine-tuned BERT model for news headline classification.

Model Export

The trained model and tokenizer were saved for future inference and deployment.

model.save_pretrained("news_classifier_tf")
tokenizer.save_pretrained("news_classifier_tf")
Streamlit Deployment

A lightweight Streamlit application was developed to provide real-time predictions.

Features
User enters a news headline.
Headline is automatically tokenized.
Model predicts the topic category.
Results are displayed instantly in the browser.
Supported Categories
World
Sports
Business
Sci/Tech

This deployment enables users to interact with the fine-tuned BERT model through a simple and user-friendly web interface.
