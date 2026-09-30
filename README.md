# AttritionAI — Employee Attrition Analysis & Prediction

## Important originality note
This project was authored from scratch for the internal hackathon request. It is **not a copied GitHub project**. The app structure, UI workflow, risk-band logic, warning-signal layer, and presentation were created specifically for this problem statement.

A bundled synthetic/demo CSV is included only so the prototype runs immediately. If your college gives you an official hackathon dataset, put it at `data/employee_attrition.csv`; the app will automatically use it.

## Run on Windows
```powershell
cd AttritionAI_Original_Hackathon
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Modules
1. Dashboard
2. Attrition Analysis
3. Model Lab
4. Risk Predictor
5. Data Explorer

## ML
- Logistic Regression
- Random Forest
- Gradient Boosting
- Decision Tree
- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC
- Stratified 80/20 hold-out test set

## Hackathon demo
Dashboard → Analysis → Model Lab → Risk Predictor → Explain warning signals.

## Responsible-use design
The probability is a statistical estimate. The product is deliberately framed as decision support and not as an automated employment decision system.
