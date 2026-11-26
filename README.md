# FinBridge Capital – Credit Risk Assessment & Default Prediction

## Live Demo
**Try the application here:**  
https://finbridge-k8vasgnlp46hqtg3rhqdrv.streamlit.app/

## Project Overview
FinBridge Capital is a machine learning-based credit risk assessment application designed to evaluate an applicant's relative risk of loan default.

The application uses historical loan data and machine learning techniques to generate risk scores and support credit assessment.

## Features
- Interactive credit risk prediction.
- Relative default risk score.
- Low, moderate, and high risk classification.
- Model performance evaluation.
- Feature importance visualization.
- Dataset exploration.
- Interactive Streamlit dashboard.

## Machine Learning
The project compares multiple machine learning models:
- Logistic Regression
- HistGradientBoosting Classifier

**Selected Model:** HistGradientBoosting

**Validation ROC-AUC:** 0.9031

## Technologies Used
- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- SciPy
- Joblib

## Project Structure
```text
FinBridge/
│
├── app.py
├── train_model.py
├── requirements.txt
├── train.csv
├── test.csv
│
├── models/
│   └── finbridge_models.pkl
│
└── outputs/
    ├── feature_importance.csv
    ├── submission.csv
    └── validation_metrics.csv
```

## Run Locally

1. Clone the repository:
```bash
git clone https://github.com/AP24110011669/FinBridge.git
```

2. Navigate to the project:
```bash
cd FinBridge
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Start the application:
```bash
streamlit run app.py
```

## Important Note
The displayed risk score is rank-scaled and is not a calibrated probability of default. The application is intended as a decision-support tool, not an automatic lending decision.

## Links
- **Live Application:** https://finbridge-k8vasgnlp46hqtg3rhqdrv.streamlit.app/
- **GitHub Repository:** https://github.com/AP24110011669/FinBridge
