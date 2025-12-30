# AquaSure – Water Potability Prediction System

AquaSure is an end-to-end machine learning application that predicts whether water is potable based on physicochemical properties.  
The project demonstrates a practical ML workflow including data preprocessing, model training, API-based inference, and Dockerized deployment.

---

## 🚀 Features Implemented

- Data preprocessing pipeline with:
  - Missing value imputation (median strategy)
  - Outlier handling using the IQR method
  - Train-test split with class stratification
- Machine learning model training (XGBoost)
- Model evaluation using standard classification metrics
- FastAPI-based inference service
- Interactive web UI for prediction (HTML + CSS)
- Dockerized application for portable deployment
- Basic CI pipeline with:
  - Code quality checks using flake8
  - Unit testing using pytest

---

## 🧠 Dataset

- **Water Potability Dataset**
- Features include:
  - pH
  - Hardness
  - Solids
  - Chloramines
  - Sulfate
  - Conductivity
  - Organic Carbon
  - Trihalomethanes
  - Turbidity
- Target variable: `Potability` (0 = Not Potable, 1 = Potable)

---

## 🛠️ Tech Stack

- **Language**: Python 3.10
- **ML**: Scikit-learn, XGBoost
- **API**: FastAPI
- **Frontend**: HTML, CSS
- **Containerization**: Docker
- **CI**: GitHub Actions (flake8 + pytest)

---

## 📁 Project Structure

```
AquaSure/
│
├── src/
│ ├── preprocessing.py
│ ├── train.py
│ └── predict.py
│
├── tests/
│ ├── test_preprocessing.py
│ └── test_model.py
│
├── data/
│ └── water_potability.csv
│
├── model/
│ └── xgboost_model.pkl
│
├── artifacts/
│ └── processed datasets
│
├── Dockerfile
├── requirements.txt
└── README.md
```
