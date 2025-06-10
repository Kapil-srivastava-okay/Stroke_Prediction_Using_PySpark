# 🧠 Stroke Prediction using Machine Learning and PySpark

This project aims to predict the likelihood of a person experiencing a stroke based on various health and demographic features. It involves data preprocessing, feature engineering, class balancing using SMOTE, and training models using both traditional machine learning and distributed computing with PySpark.

---

## 📁 Dataset

The dataset used is the **Stroke Prediction Dataset** from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset). It consists of 5,110 records with the following features:

- **Demographics**: `gender`, `age`, `ever_married`, `work_type`, `Residence_type`
- **Health indicators**: `hypertension`, `heart_disease`, `avg_glucose_level`, `bmi`, `smoking_status`
- **Target variable**: `stroke` (1 = stroke occurred, 0 = no stroke)

---

## 🔍 Feature Extraction

The following steps were taken during preprocessing:

- Handled missing values in the `bmi` column.
- Encoded categorical variables using Label Encoding and One-Hot Encoding.
- Normalized numerical features like `avg_glucose_level` and `bmi`.
- Constructed a final feature matrix for model training.

---

## ⚖️ SMOTE (Synthetic Minority Over-sampling Technique)

The dataset is highly imbalanced with only about 5% positive stroke cases. To mitigate this:

- Applied SMOTE to oversample the minority class in the training data.
- Helped improve recall and overall model balance without sacrificing too much precision.

---

## ⚙️ PySpark Integration

To enable distributed processing and scalability:

- The dataset was converted into a **PySpark DataFrame**.
- Used `VectorAssembler` to create feature vectors.
- Trained a `RandomForestClassifier` using PySpark MLlib.
- Built a streamlined pipeline for feature transformation and modeling.

---

## 📊 Results

Model performance was assessed using:

- **Accuracy**, **Precision**, **Recall**, **F1-Score**
- **Confusion Matrix**
- **ROC-AUC Curve**

### Highlights:
- SMOTE improved the model’s sensitivity to the minority class.
- PySpark enabled scalable training for larger datasets.
- The final model achieved a balanced trade-off across key performance metrics.

---

## 🚀 Technologies Used

- Python (Pandas, NumPy, Scikit-learn)
- PySpark (MLlib, DataFrame API)
- Imbalanced-learn (SMOTE)
- Matplotlib, Seaborn (visualizations)

---

## 📌 Future Work

- Hyperparameter tuning using GridSearchCV or Spark's `CrossValidator`.
- Deployment of the trained model as a REST API.
- Incorporation of time-series or patient history data for improved prediction accuracy.
