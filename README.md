# 🏠 House Price Prediction | Kaggle ML Competition

A machine learning project built for the Kaggle **House Prices: Advanced Regression Techniques** competition. This project uses a **Random Forest Regressor** to predict house sale prices from structured housing data, following a complete end-to-end machine learning workflow.

The focus is on feature selection, model training, validation using **Mean Absolute Error (MAE)**, and generating Kaggle-ready submission files.

---

## ✨ Features

* Structured data loading and preprocessing using Pandas
* Feature selection based on housing attributes
* Train-validation split for model evaluation
* Random Forest regression model
* Model performance evaluation using MAE
* Generation of competition submission file

---

## 🛠 Tech Stack

* **Python**
* **Pandas** – Data manipulation
* **scikit-learn** – Model training & evaluation
* **Random Forest Regressor** – Prediction model

---

## 📂 Dataset

* **Competition:** House Prices – Advanced Regression Techniques
* **Source:** Kaggle
* **Files Used:**

  * `train.csv` – Training data
  * `test.csv` – Test data

---

## 🚀 How to Run Locally

1. Clone the repository

   ```bash
   git clone <repo-url>
   ```
2. Install dependencies

   ```bash
   pip install pandas scikit-learn
   ```
3. Place `train.csv` and `test.csv` in the project directory
4. Run the script

   ```bash
   python main.py
   ```
5. A `submission.csv` file will be generated for Kaggle upload

---

## 📊 Model Evaluation

* **Model:** Random Forest Regressor
* **Metric:** Mean Absolute Error (MAE)
* **Validation Strategy:** Train / Validation split

---

## 📌 Project Type

> **Machine learning learning project** focused on understanding regression models, evaluation metrics, and competition-style ML workflows.

---

## 🚧 Future Improvements

* Feature engineering and handling missing values
* Hyperparameter tuning
* Cross-validation
* Comparison with other regression models (XGBoost, Gradient Boosting)

---
