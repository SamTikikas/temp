# Mobile Diabetic App

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Diabetes Prediction Using Machine Learning](#diabetes-prediction-using-machine-learning)
  - [Objective](#objective)
  - [Dataset](#dataset)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Training](#model-training)
  - [Backend](#backend)
  - [Deployment](#deployment)
  - [Front-End UI](#front-end-ui)
- [Diabetic Retinopathy Detection](#diabetic-retinopathy-detection)
  - [Objective](#objective-1)
  - [Dataset](#dataset-1)
  - [Data Preprocessing](#data-preprocessing-1)
  - [Model Training](#model-training-1)
  - [Evaluation Metrics](#evaluation-metrics)
  - [Front-End UI](#front-end-ui-1)
- [Results](#results)
- [How to Run](#how-to-run)
- [References](#references)

## Introduction
The **Mobile Diabetic App** is an advanced health application designed to assist individuals in predicting diabetes and detecting diabetic retinopathy using machine learning models. The app provides a user-friendly interface built with Flutter and integrates powerful machine learning algorithms for accurate predictions.

**Mentor:** Dr. Devika Rubi

## Features
- Diabetes prediction using ML models (Decision Tree, Random Forest, XGBoost, SVM)
- Diabetic retinopathy detection using CNN-based deep learning
- User-friendly mobile interface built with Flutter
- Real-time prediction with Flask API
- Secure authentication using Firebase Firestore
- Deployed on AWS EC2 and Google Cloud Platform

## Technology Stack
- **Frontend:** Flutter (Dart)
- **Backend:** Flask (Python)
- **Database:** Firestore
- **Machine Learning Models:** Scikit-Learn, XGBoost, TensorFlow
- **Deployment:** AWS EC2, GCP App Engine

## Diabetes Prediction Using Machine Learning
### Objective
The goal of this module is to develop predictive models to identify whether a person has diabetes based on diagnostic features.

### Dataset
- **Source:** [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Features:** Number of pregnancies, BMI, glucose levels, insulin levels, age, etc.
- **Target:** Binary classification (Diabetic/Non-Diabetic)

### Data Preprocessing
- Handling missing values (replacing zeros with NaN)
- Feature scaling using StandardScaler
- Splitting data into training and testing sets (67% training, 33% testing)

### Model Training
- Algorithms used: Decision Tree, Random Forest, XGBoost, SVM
- Random Forest achieved the highest accuracy (77.16%)
- Trained using AWS SageMaker

### Backend
- Flask framework handles API requests at port 5000
- User authentication and data storage managed using Firebase Firestore

### Deployment
- Flask API hosted on **Google Cloud Platform (GCP) App Engine**
- Also deployed on **AWS Elastic Compute Cloud (EC2)**

### Front-End UI
- Built using Flutter
- Real-time prediction interface with data validation and visualization
- Interactive alert dialogues for risk assessment

## Diabetic Retinopathy Detection
### Objective
To detect **diabetic retinopathy** in patients using retinal images and a CNN-based deep learning model.

### Dataset
- **Source:** [Diabetic Retinopathy Detection Dataset](https://www.kaggle.com/competitions/diabetic-retinopathy-detection)
- **Classes:** 0 - No DR, 1 - DR detected
- **Preprocessed images:** Normalized and resized for consistency

### Data Preprocessing
- Convert images to NumPy arrays
- Normalize pixel values (dividing by 255.0)

### Model Training
- Convolutional Neural Network (CNN) trained using Keras & TensorFlow
- Activation functions: Sigmoid, ReLU
- Optimizer: ADAM
- Loss function: Binary Cross-Entropy
- Trained on **80% dataset**, tested on **20% dataset**

### Evaluation Metrics
- Confusion Matrix
- Precision, Recall, F1-Score, Accuracy

### Front-End UI
- Select and display retinal images
- Toggle switch for ground truth visibility
- Sends image to Flask server for real-time prediction

## Results
Below are the screenshots of the results obtained from model predictions and application usage:

*(Upload your screenshots here)*

## How to Run
### Prerequisites
- Python 3.8+
- Flutter installed
- Firebase setup
- AWS/GCP account for deployment

### Steps
1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/mobile-diabetic-app.git
   cd mobile-diabetic-app
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   flutter pub get
   ```
3. Run the backend server:
   ```sh
   python app.py
   ```
4. Start the Flutter app:
   ```sh
   flutter run
   ```


### Results


## References
- [Diabetes Prediction Using Machine Learning](https://pubmed.ncbi.nlm.nih.gov/31518657/)
- [Diabetic Retinopathy Dataset](https://www.kaggle.com/competitions/diabetic-retinopathy-detection)
- [Convolutional Neural Networks](https://towardsdatascience.com/convolutional-neural-networks-explained-9cc5188c4939)
