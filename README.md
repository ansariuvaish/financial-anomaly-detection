
# Financial Anomaly Detection Using Statistical Machine Learning Methods

## 📌 Project Overview

Financial fraud is a major challenge in digital payment systems. Fraudulent credit-card transactions are rare compared to legitimate transactions, making them difficult to detect using traditional approaches.

This project uses statistical methods and machine learning anomaly detection techniques to identify suspicious credit-card transactions. The detectors are trained using normal transactions only, following a one-class anomaly detection approach.

The goal is to identify potentially fraudulent transactions while minimizing false alarms.

---

## 🎯 Objectives

- Analyze credit-card transaction data.
- Understand the characteristics of fraudulent and legitimate transactions.
- Apply statistical anomaly detection techniques.
- Implement machine learning-based anomaly detectors.
- Evaluate models using suitable metrics for imbalanced data.
- Compare models and analyze their performance.
- Develop a dashboard for visualizing anomaly detection results.

---

## 📊 Dataset

**Dataset:** ULB Credit Card Fraud Detection Dataset (Kaggle)

- Total transactions: 284,807
- Fraudulent transactions: 492
- Fraud percentage: Approximately 0.17%
- Features: V1–V28 (anonymized PCA components), Time, and Amount

The dataset is highly imbalanced, with fraudulent transactions representing a very small proportion of all transactions.

---

## ⚙️ Methodology

The project follows these stages:

1. Data loading and inspection
2. Exploratory Data Analysis (EDA)
3. Data preprocessing
4. Leakage prevention and validation strategy
5. Statistical anomaly detection
6. Machine learning anomaly detection
7. Threshold selection using validation data
8. Model evaluation on test data
9. Error analysis
10. Dashboard development and visualization

---

## 🧪 Statistical Methods

The following statistical approaches are implemented as baseline anomaly detectors:

- Z-Score
- Interquartile Range (IQR)
- Mahalanobis Distance

These methods help identify transactions that deviate significantly from normal transaction patterns.

---

## 🤖 Machine Learning Models

The following anomaly detection models are evaluated:

- PCA Reconstruction Error
- Gaussian Mixture Model (GMM)
- Local Outlier Factor (LOF)
- One-Class Support Vector Machine (One-Class SVM)
- Isolation Forest

The models are trained using normal transactions only, without using fraud labels during fitting.

---

## 📈 Evaluation Metrics

Due to the highly imbalanced nature of the dataset, the project uses multiple evaluation metrics:

- Precision
- Recall
- F1-Score
- Area Under the Precision-Recall Curve (AUPRC)
- Confusion Matrix

Thresholds are selected using validation data to balance detection performance and false alarms.

---

## 🏆 Highlighted Result

The highlighted model in our evaluation is PCA Reconstruction Error using the smallest-variance component.

| Metric | Result |
|---|---:|
| Test AUPRC | 0.671 |
| F1-Score | 0.807 |
| Precision | 0.85 |
| Recall | 0.77 |

The threshold was selected using validation data.

*These are the current project results and may be updated as the final analysis is consolidated.*

---

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook
- Streamlit (Dashboard)

---

## 📁 Repository Structure

```text
financial-anomaly-detection/
│
├── Financial_Anomaly_Detection.ipynb
├── README.md
├── LICENSE
├── .gitignore
│
├── data/
│
├── src/
│
├── results/
│
├── dashboard/
│
└── report/
```

The repository structure will be expanded as development progresses.

---

## 👥 Team Members

| Name | Roll Number | Role |
|---|---|---|
| Uvaish | S25CSEU2265 | Project Lead, ML Detectors, Validation, Evaluation, Error Analysis |
| Amaan Ullah | S25CSEU2255 | Data, EDA, Preprocessing, Leakage Controls |
| Harsh Sharma | S25CSEU2252 | Web Dashboard, Documentation, Report and Presentation |

---

## 🚀 Future Scope

- Develop an interactive dashboard for transaction analysis.
- Improve anomaly detection performance.
- Explore additional feature engineering techniques.
- Compare additional statistical and machine learning methods.
- Develop a real-time transaction monitoring prototype.
- Improve model interpretability and explainability.

---

## ⚠️ Disclaimer

This project is developed for academic and educational purposes. Anomaly detection results indicate potentially suspicious transactions and should not be considered definitive proof of fraud.

---

## 📄 License

This project is licensed under the MIT License.
