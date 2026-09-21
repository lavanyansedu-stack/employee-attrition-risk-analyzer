# AI-Powered Employee Attrition Risk Analyzer & HR Decision Support Dashboard

## 1. Project objective

This project analyzes employee data, identifies historical attrition patterns, predicts a probability-based attrition risk score, explains potential contributing signals, and presents HR analytics through Streamlit.

It is a **prototype HR decision-support system**. It must not be presented as a system that can know whether an employee will resign.

## 2. Dataset

Use the IBM HR Analytics Employee Attrition & Performance dataset described in the project specification.

Download the CSV and rename it:

`employee_attrition.csv`

Place it here:

`data/employee_attrition.csv`

Expected source file:

`WA_Fn-UseC_-HR-Employee-Attrition.csv`

## 3. Architecture

```text
CSV Dataset
    |
    v
preprocessing.py
    |
    +--> cleaning
    +--> feature engineering
    +--> one-hot encoding
    +--> train/test split
    |
    v
train_model.py
    |
    +--> Logistic Regression
    +--> Decision Tree
    +--> Random Forest
    +--> Accuracy / Precision / Recall / F1 / ROC-AUC
    |
    v
models/attrition_model.pkl
    |
    +--> predict.py
    +--> explainability.py
    +--> recommendations.py
    |
    v
app/app.py
    |
    +--> Executive Dashboard
    +--> HR Analytics
    +--> Employee Risk Predictor
    +--> Risk Analysis
    +--> HR Insights
```

## 4. Installation

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

Install packages:

```bash
pip install -r requirements.txt
```

## 5. Train the models

From the project root:

```bash
python -m src.train_model
```

This compares:

- Logistic Regression
- Decision Tree
- Random Forest

The final model is selected using recall first, then F1 and ROC-AUC, because the project is intended as an early-warning system.

Generated files:

```text
models/
├── attrition_model.pkl
├── model_metrics.json
└── feature_info.json
```

## 6. Run Streamlit

```bash
streamlit run app/app.py
```

## 7. Risk bands

The project uses demonstration thresholds:

- 0–30%: Low Risk
- 31–60%: Medium Risk
- 61–100%: High Risk

These are project-defined thresholds, not scientifically validated HR standards.

## 8. Important responsible-AI note

The model is based on a fictional IBM HR dataset. It should be demonstrated as an educational prototype, not as a production system trained on confidential employee records.

Do not say:

> "This employee will resign."

Use:

> "These factors contributed to the model's elevated risk score and may be useful areas for HR review."

The model should support human review rather than automatically making employment decisions.

## 9. Optional SHAP

`shap` is included in requirements for an advanced explainability extension. The current dashboard provides transparent rule-based review signals and global model feature importance. SHAP can be added after the basic pipeline is working.
