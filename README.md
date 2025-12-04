# ✈️ Airline Passenger Satisfaction Prediction using Artificial Neural Networks (ANN)
📌 Project Overview

Airline industries receive massive customer feedback daily, but manually analyzing satisfaction levels is inefficient.
This project builds a Deep Learning–based Passenger Satisfaction Prediction System using Artificial Neural Networks (ANN) to automatically classify whether a passenger is Satisfied or Not Satisfied.

The solution includes:

Complete EDA

Feature Engineering and Preprocessing Pipelines

ANN Model Development with Hyperparameter Tuning (HyperBand)

Saving Model + Pipeline (H5 + PKL)

Designing a fully interactive Streamlit Web Application

Deployment-ready architecture

# 🎯 Problem Statement

Airlines want to improve customer experience by understanding what factors influence passenger satisfaction.
Given various flight attributes such as:

Passenger details

Flight delay information

Service ratings (Food quality, Cleanliness, Comfort, etc.)

Travel type and class

Predict whether the passenger was satisfied or dissatisfied.

# 🛠 Solution Approach
# 1️⃣ Data Understanding & Cleaning

* The dataset (Airline Passenger Satisfaction) contains:

* Numerical features

* Categorical features

* Service rating features

* Delay metrics

* Performed:

* Missing value analysis

* Outlier detection

* Feature distribution study

* Correlation analysis (univariate & bivariate)

# 🔍 2️⃣ Exploratory Data Analysis (EDA)

* EDA included:

* Distribution plots for numerical features

* Count plots for categorical features

* Boxplots for outliers

* Correlation heatmaps

* Relationship analysis between satisfaction and each feature

* Insights found:

* Business class passengers show higher satisfaction

* Longer delays reduce satisfaction

* Service quality ratings highly correlate with satisfaction

* Loyal customers are mostly satisfied

# 🧩 3️⃣ Feature Engineering

* Applied extensive feature engineering using Scikit-Learn Pipelines:

* Categorical Features

* Label Encoding (Gender, Customer Type)

* Ordinal Encoding (Class → Eco < Eco Plus < Business)

* OneHot Encoding (multi-class fields)

* Numerical Features

* KNN Imputer for missing values

* StandardScaler for normalization

* Custom Transformer

* Implemented LabelEncoderTransformer for deployment compatibility

* Pipeline

* A complete ColumnTransformer + Pipeline was created to automate preprocessing during training & deployment.

# 🤖 4️⃣ Model Building — ANN

* Built a deep learning classification model with:

* Input layer based on transformed features

* Multiple hidden layers

* ReLU activation

* He-normal initialization

* Batch Normalization

* Dropout

* L1 Regularization for feature sparsity

* Sigmoid output layer

## Compiled with:
optimizer = Adam(learning_rate)
loss = "binary_crossentropy"
metrics = ["accuracy"]

# 🔧 5️⃣ Hyperparameter Tuning — HyperBand (Keras Tuner)

* Used Keras Tuner (HyperBand) to search optimal values for:

* Number of layers

* Number of neurons per layer

* Dropout rate

* Learning rate

* L1 regularization strength

Best accuracy reached: ~95.3% validation accuracy

# 💾 6️⃣ Saving Model

* ANN model saved as: best_airline_ann_model.h5

* Preprocessing pipeline saved as: airline_preprocessor_pipeline.pkl

* These files are used directly in Streamlit for predictions.

# 🌐 7️⃣ Streamlit Web App

* A modern, animated UI with:

* ✔ Star-field animated background
* ✔ 3D floating title
* ✔ Glass-card design
* ✔ Slider inputs (0–5 ratings)
* ✔ Auto preprocessing via saved PKL
* ✔ Real-time ANN predictions

### The app collects:

* Passenger info

* Flight metrics

* Service ratings

### Outputs:

* Satisfied 😃

* Not Satisfied 😞
* With prediction confidence score.

# 🚀 8️⃣ Deployment

### Ready for deployment on:

* Streamlit Cloud

* Render

* HuggingFace Spaces

* Local Hosting

### Uses:

* requirements.txt

* .streamlit/config.toml

* Python 3.11 compatibility

# 🧪 9️⃣ Results

* Training Accuracy: High (after tuning)

* Validation Accuracy: ~95.3%

* Model generalizes well with no overfitting

* Most impactful features:

      * Inflight entertainment
      
      * Online support
      
      * Seat comfort
      
      * Class
      
      * Loyalty status
# 📂 Project Structure

📁 Airline-Satisfaction-Prediction-ANN
│── app.py
│── best_airline_ann_model.h5
│── airline_preprocessor_pipeline.pkl
│── requirements.txt
│── README.md
│── notebook.ipynb (Jupyter analysis)
│── dataset.csv

# 🧰 Tech Stack

Python, Pandas, NumPy, Scikit-Learn, TensorFlow/Keras, Keras Tuner (HyperBand), ANN, EDA, Feature Engineering, Pipelines, Joblib, Streamlit, GitHub, Deployment

# 🌟 Key Features

* Automated end-to-end ML pipeline

* Fully tuned ANN model

* Highly interactive UI

* Deployment-ready architecture

* Feature engineering optimized for real-world use

* Modern UI (3D title, animated background, glass cards)
