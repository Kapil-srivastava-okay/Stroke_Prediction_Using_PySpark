# Stroke Prediction using PySpark and Tableau

This project applies **machine learning techniques** and **data visualization** to predict the occurrence of strokes using demographic and health data. The primary model used is **Random Forest**, implemented with **PySpark**, with visual insights generated using **Tableau**.

## 📁 Project Overview

- **Objective**: Predict the likelihood of a stroke using a healthcare dataset.
- **Dataset**: [Stroke Prediction Dataset on Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- **Technologies Used**:  
  - PySpark (MLlib)  
  - SMOTE for class imbalance  
  - Tableau  
  - Pandas, Seaborn, Matplotlib

---

## 🧠 Key Features

- **Data Preprocessing**:  
  - Imputation of missing BMI values  
  - Type casting of numerical and categorical columns  
  - Encoding of categorical features

- **Machine Learning with PySpark**:  
  - Random Forest Classifier  
  - Feature importance analysis  
  - Model evaluation with ROC Curve, Confusion Matrix, Accuracy, Precision, Recall, and F1 Score

- **Class Imbalance Handling**:  
  - Synthetic Minority Oversampling Technique (SMOTE)

- **Data Visualization**:  
  - Tableau dashboards for age, glucose levels, gender, BMI, smoking status, and stroke occurrence  
  - Heatmaps, treemaps, histograms, line graphs, and box plots

---

## 🛠 Setup & Installation

1. Install and configure Apache Spark and Hadoop locally.
2. Install required Python libraries:

```bash
pip install pandas matplotlib seaborn findspark imbalanced-learn
