# Dashboard

Interactive Streamlit dashboard for the **Financial Anomaly Detection** project (Statistical Machine Learning Lab). It lets you explore how 8 anomaly detectors perform on credit-card fraud data and score new transactions.

## Pages

| Page | What it shows |
|------|---------------|
| **Dashboard** | Headline numbers, AUPRC comparison of all 8 methods, fraud vs normal chart, sortable leaderboard |
| **Model Performance** | Precision-recall and ROC curves, with methods you can add or remove |
| **Threshold Analysis** | Slider for the number of alerts, with the confusion matrix, precision, recall and F1 |
| **Live Transaction Scoring** | Upload a CSV (or load the demo file), rank transactions by anomaly score, download results |
| **Methodology** | Dataset, preprocessing, methods, evaluation and limitations |

## Key results (held-out test set, 85,118 transactions, 142 frauds)

Best detector: **PCA reconstruction error**. AUPRC 0.671, precision 85%, recall 77%. It caught 109 of 142 frauds with 19 false alarms.

## Run locally

```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

`app.py` expects these files next to it:

- `data/dashboard_data.json`
- `data/demo_transactions.csv`
- `models/bundle.joblib`

## CSV format for Live Transaction Scoring

- Required columns: `V1` to `V28` and `Amount` (numeric, no missing values, `Amount` not negative)
- Optional columns: `Time` (ignored) and `Class` (0 = normal, 1 = fraud)
- Up to 100,000 rows

## Notes

- The anomaly score is the detector's raw output. It is **not** a fraud probability.
- The demo file has about 1.5% fraud, roughly 9x the real rate, so its precision is not a real-world estimate.

## Technology

Python, Streamlit, Pandas, NumPy, SciPy, Scikit-learn, Plotly, Joblib.

## Team

Uvaish, Amaan Ullah, Harsh Sharma
