# 🚀 Machine Learning Projects Portfolio

This repository contains three machine learning and deep learning projects covering customer churn prediction, multimodal learning, and natural language processing using transformer-based models.

---

# 📌 Task 1: End-to-End ML Pipeline with Scikit-learn

## 🎯 Objective

Develop a reusable and production-ready machine learning pipeline to predict customer churn using the Telco Customer Churn dataset.

## 📂 Dataset

**Dataset:** Telco Customer Churn

The dataset contains customer demographic information, service subscriptions, account details, and customer behavior records used to predict customer churn.

## ⚙️ Data Preprocessing

The following preprocessing steps were performed:

* Removed missing values and duplicate records.
* Applied One-Hot Encoding to categorical features.
* Standardized numerical features using StandardScaler.
* Integrated preprocessing using ColumnTransformer.
* Built an end-to-end Scikit-learn Pipeline for automated data processing.

## 🤖 Model Development

Two classification models were implemented:

### Logistic Regression

* Linear classification model
* Efficient and interpretable
* Suitable for binary classification tasks

### Random Forest Classifier

* Ensemble learning algorithm
* Handles non-linear relationships effectively
* Reduces overfitting through multiple decision trees

Both models were integrated into complete machine learning pipelines for preprocessing, training, and evaluation.

## 🔍 Hyperparameter Tuning

GridSearchCV was used to optimize model performance.

### Logistic Regression

* Penalty types
* Solver configurations
* Regularization strengths (C)

### Random Forest

* Number of estimators
* Maximum tree depth
* Feature selection strategies

## 📊 Model Performance

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 81.9%    |
| Random Forest       | 73.5%    |

### Key Insight

✅ Logistic Regression achieved better performance and demonstrated stronger generalization on unseen data.

## 💾 Model Export

The trained models were saved using Joblib:

* `logistic_regression_churn.pkl`
* `random_forest_churn.pkl`

---

# 🏠 Task 2: Multimodal Housing Price Prediction (Images + Tabular Data)

## 🎯 Objective

Develop a multimodal deep learning model capable of predicting house prices by combining image data with structured tabular features.

## 📂 Dataset

### Structured Data

Housing sales dataset containing:

* Square Footage
* Bedrooms
* Bathrooms
* City Codes

### Image Data

Property images stored in the custom dataset:

`socal_pics/`

Each property record is linked to its corresponding image through a unique `image_id`.

## ⚙️ Data Preparation

* Created an `image_path` column for image mapping.
* Split dataset into training (80%) and validation (20%) sets.
* Selected key numerical features:

  * sqft
  * bed
  * bath
  * n_citi
* Applied ImageDataGenerator for image normalization and batching.

## 🧠 Feature Fusion

A custom generator was developed to synchronize image and tabular inputs.

### Image Branch (CNN)

* Convolution Layers
* ReLU Activation
* Max Pooling Layers
* Flatten Layer

### Tabular Branch

* Dense Layers
* Batch Normalization
* Dropout Regularization

### Fusion Layer

Both feature streams were merged using a Concatenate layer before final price prediction.

## 🏗️ Model Configuration

* Optimizer: Adam
* Loss Function: Mean Squared Error (MSE)
* Metrics:

  * Mean Absolute Error (MAE)
  * Root Mean Squared Error (RMSE)

### Callbacks

* EarlyStopping
* ReduceLROnPlateau

### Training Parameters

* Epochs: 15
* Batch Size: 64
* Image Size: 128 × 128

### Model Parameters

* Total Parameters: ~4.26 Million
* Trainable Parameters: ~4.26 Million

## 📊 Results

| Metric | Training  | Validation |
| ------ | --------- | ---------- |
| MAE    | ~696K USD | ~703K USD  |
| RMSE   | ~785K USD | ~796K USD  |

## 📈 Visualization

RMSE curves for training and validation datasets were plotted across epochs. Results indicate stable convergence and minimal overfitting, demonstrating effective multimodal feature learning.

---

# 📰 Task 3: Fine-Tuning BERT for News Topic Classification

## 🎯 Objective

Fine-tune a BERT-based transformer model to classify news headlines into predefined topic categories.

## 📂 Dataset

**Dataset:** AG News (Hugging Face Datasets)

The dataset contains thousands of news headlines categorized into:

* World
* Sports
* Business
* Science & Technology

## 🛠️ Technologies Used

* TensorFlow
* Hugging Face Transformers
* Streamlit

## ⚙️ Data Preprocessing & Tokenization

* Loaded dataset using the Hugging Face Datasets library.
* Tokenized text using `bert-base-uncased`.
* Applied padding and truncation with a maximum sequence length of 128 tokens.
* Converted data into TensorFlow datasets using `to_tf_dataset()`.

## 🤖 Model Fine-Tuning

A `TFBertForSequenceClassification` model was initialized with four output classes.

### Training Configuration

* Optimizer: Adam
* Learning Rate: 2e-5
* Loss Function: SparseCategoricalCrossentropy
* Metric: Accuracy
* Epochs: 2

## 📊 Model Evaluation

Performance was evaluated using:

* Accuracy Score
* Weighted F1-Score

The fine-tuned BERT model achieved strong performance in classifying news headlines across all categories.

## 💾 Model Export

The trained model and tokenizer were saved for future deployment and inference:

```python
model.save_pretrained("news_classifier_tf")
tokenizer.save_pretrained("news_classifier_tf")
```

## 🌐 Streamlit Deployment

A Streamlit web application was developed to provide real-time predictions.

### Features

* User enters a news headline.
* Text is automatically tokenized.
* BERT model predicts the topic category.
* Prediction is displayed instantly.

### Supported Categories

* World
* Sports
* Business
* Sci/Tech

This deployment enables users to interact with the fine-tuned model directly through a user-friendly web interface.
