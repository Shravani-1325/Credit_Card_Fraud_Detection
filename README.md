# 💳 Credit Card Fraud Detection using Machine Learning

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Project-green)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-red)
![Dataset](https://img.shields.io/badge/Dataset-Kaggle-blue)

---

## 📊 1. Project Overview

This project addresses the "needle in a haystack" challenge of credit card fraud detection. Using a dataset of **284,807 European transactions**, by leveraging machine learning to identify fraudulent activity within a highly imbalanced environment where fraud accounts for only **0.17%** of the data.

---

## 📉 2. Problem Statement

The scale of digital payments makes manual fraud review impossible, yet the financial stakes are high:

* **False Negatives:** Undetected fraud leads to direct financial loss.
* **False Positives:** Incorrectly flagged transactions erode customer trust.

**The Goal:** Build a robust classifier using **SMOTE** and **threshold tuning** to maximize **Recall** (catching fraud) while maintaining high **Precision** (minimizing "false alarms").

---
## 🎯 3. Project Objectives

The primary goal is to develop a robust ML pipeline to identify fraudulent transactions within a highly imbalanced environment (**0.17%** fraud).

> ### Technical Goals:

* **Class Imbalance:** Implement **SMOTE** and Under-sampling to ensure the model learns rare fraud patterns.
* **Model Optimization:** Train and fine-tune classifiers (Random Forest, XGBoost) using **Cross-Validation** to ensure generalizability.
* **Threshold Tuning:** Adjust classification thresholds to balance the "cost of missed fraud" against "customer friction."
* **Performance Evaluation:** Prioritize **Recall, Precision, F1-score, and AUPRC** over standard accuracy.

---

## 📊 4. Data Understanding

The project utilizes the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle, representing 48 hours of transaction activity.

> ### **Dataset Characteristics**

| Feature | Count / Value |
| --- | --- |
| **Total Transactions** | 284,807 |
| **Fraudulent (Class 1)** | 492 (**0.172%**) |
| **Legitimate (Class 0)** | 284,315 |


> ### **Feature Breakdown**
* The dataset contains `30 features` :
* **V1 – V28:** Anonymized features via **PCA transformation**.
* **Time:** Seconds elapsed since the first transaction.
* **Amount:** The transaction value.
* **Class (Target):** `0` (Legitimate) vs. `1` (Fraudulent).

---


## 🔍 5. Exploratory Data Analysis (EDA)

EDA was conducted to uncover fraud patterns and distribution shifts between legitimate and fraudulent classes.

> ### **Key Observations**

* **Severe Imbalance:** Fraud cases account for only **0.172%** of the data.
* **Skewed Distribution:** Transaction amounts are highly skewed; most fraud occurs at lower amounts, but significant outliers exist.
* **Feature Variance:** PCA features like $V17$, $V14$, and $V12$ show strong negative correlations with fraud, while $V11$ and $V4$ show positive correlations.

> ### **Core Visualizations**

1. **Class Distribution:** Bar and Pie charts illustrating the extreme 99.8% vs 0.2% split.
2. **Amount Analysis:** Density plots comparing Fraud vs. Legitimate transaction values.
3. **Log Transformation:** Applied to the `Amount` feature to normalize the distribution for better visualization.
4. **Time Series Analysis:** Hourly fraud frequency analysis to detect cyclical patterns.
5. **Correlation Heatmap:** Identifying which anonymized $V$ features are most indicative of fraud.

---

## ⚙️ 6. Data Preprocessing

To ensure optimal model performance and handle the skewness of raw features, the following pipeline was implemented:

 ### **1. Data Integrity**

* **Cleaning:** Confirmed zero missing values across all 284,807 rows.
* **Consistency:** Verified that PCA features were already centered and scaled.

 ### **2. Feature Engineering & Scaling**

* **Log Transformation:** Applied to the `Amount` feature to normalize its highly skewed distribution and reduce the influence of extreme outliers.
* **Standardization:** Utilized `StandardScaler` to bring `Time` and `Amount` into the same numerical range as the V1–V28 features, preventing any single variable from dominating the model.

 ### **3. Data Splitting**

* **Stratified Shuffle Split:** Divided data into **Training** and **Testing** sets while strictly preserving the **0.17% fraud ratio** in both. This ensures the model is evaluated on a representative sample of rare events.

---

## ⚖️ 7. Handling Class Imbalance

To prevent the models from being biased toward the majority class (Legitimate transactions), **SMOTE (Synthetic Minority Over-sampling Technique)** was implemented.

> ### **The SMOTE Approach**

Unlike simple oversampling, which duplicates existing data, SMOTE creates **synthetic examples** by interpolating between existing minority instances. This expands the decision boundary of the fraud class without causing the overfitting typically associated with data replication.

> ### Benefits:

- Improves model ability to detect fraud
- Reduces bias toward majority class
- Improves recall for fraudulent transactions
---

## 🤖 8. Machine Learning Models

The following models were evaluated using Stratified K-Fold Cross Validation (5 folds):

> ### **Model Comparison**

| Model | Primary Role | Key Advantage |
| --- | --- | --- |
| **Logistic Regression** | Baseline | High interpretability; provides a solid performance floor. |
| **Random Forest** | Robustness | Reduces overfitting through bagging; handles non-linear PCA features well. |
| **XGBoost** | Performance | State-of-the-art gradient boosting; highly effective at identifying rare patterns. |
| **SVM** | Decision Boundary | Effective in high-dimensional spaces to separate fraud from legitimate cases. |

---


## 📏 9. Model Evaluation Metrics


>### **Metric Breakdown**

| Metric | Business Focus | Technical Definition |
| --- | --- | --- |
| **Precision** | **Customer Friction** | Out of all transactions flagged as fraud, how many were actually fraudulent? |
| **Recall** | **Financial Loss** | Out of all actual fraud cases, how many did the model successfully detect? |
| **F1-Score** | **The Balance** | The harmonic mean of Precision and Recall; used to find the "sweet spot" between the two. |
| **ROC-AUC** | **Separation Power** | Measures the model's ability to distinguish between the two classes across all thresholds. |



> ### **Why Recall Matters Here**

In credit card fraud, a **False Negative** (missing a theft) is usually more expensive for a bank than a **False Positive** (a temporary card block for a legitimate user). Therefore, primary goal is to maximize **Recall** without letting **Precision** drop to an unusable level.

---


## 📊 10. Model Performance

The table below summarizes the results. We prioritized Recall (catching fraud) and F1-Score (overall balance)

| Rank | Model | F1-Score | Precision | Recall | ROC-AUC | Confusion Matrix |
| --- | --- | --- | --- | --- | --- | --- |
| **1** | **Ensemble (RF+XGB)** | **0.860** | **0.935** | 0.797 | **0.984** | `[113715, 11 / 40, 157]` |
| **2** | **RF Tuned** | 0.852 | 0.923 | 0.792 | 0.937 | `[113713, 13 / 41, 156]` |
| **3** | **RF + SMOTE** | 0.840 | 0.870 | **0.812** | 0.979 | `[113702, 24 / 37, 160]` |
| **4** | **XGBoost Tuned** | 0.821 | 0.829 | **0.812** | 0.979 | `[113693, 33 / 37, 160]` |

> ### **🏆 Summary**

* **Best Overall:** The **Ensemble Model** provides the best balance (F1=0.860) and highest separation power (AUC=0.984), with only **11 false alarms**.
* **Best Detection:** **RF + SMOTE** achieved the highest **Recall (0.812)**, catching the most fraud cases (160).
* **Conclusion:** The Ensemble is ideal for minimizing customer friction, while SMOTE-based models prioritize maximum fraud capture.

---

## 🎖️ 11. Final Model Selection

The **Random Forest + SMOTE** model was chosen for final deployment due to its superior performance in high-stakes fraud detection.

> ### **Selection Rationale**

* **Maximized Recall (0.812):** Captures more actual fraud cases than non-SMOTE models.
* **Optimal Balance:** Achieves a strong **0.840 F1-score**, balancing detection power with precision.
* **High Reliability:**  **0.979 ROC-AUC** ensures excellent class separation.


---

## 💾 12. Model Saving

The final **Random Forest + SMOTE** model was serialized using **Joblib** to enable efficient deployment and future inference without the need for retraining.

### **Usage**

```python
import joblib

# Save the model
joblib.dump(final_model, "fraud_detection_model.pkl")

# Load the model later
loaded_model = joblib.load("fraud_detection_model.pkl")

```


## 📂 13. Project Structure

```text
Credit-Card-Fraud-Detection
│
├── data
│   └── creditcard.csv             # Raw dataset (Kaggle)
│
├── notebooks
│   └── credit-card-fraud-detection.ipynb      # EDA, Preprocessing, and Modeling
│
├── README.md                      # Project documentation
└── requirements.txt               # Dependencies for reproduction

```

## 🚀 14. Getting Started

1. **Clone the repository:**
```bash
git clone https://github.com/Shravani-1325/Credit_Card_Fraud_Detection.git
```


2. **Install dependencies:**
```bash
pip install -r requirements.txt

```


3. **Run the analysis:**
Open `notebooks/credit-card-fraud-detection.ipynb` in Jupyter or VS Code to see the full pipeline from EDA to Model Saving.

---

## 👩‍💻 15. Author

Shravani More
