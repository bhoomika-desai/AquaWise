
# AquaWise: Clean Water Access Forecaster

AquaWise is a machine learning project that predicts the percentage of a population with access to safely managed drinking water using socioeconomic indicators. This project aligns with SDG 6 (Clean Water and Sanitation) and includes a full data pipeline, exploratory data analysis, model development, model saving, and a Streamlit-based user interface.

---

## Overview

The goal of this project is to forecast the availability of safely managed drinking water at the national level. Using GDP per capita and infrastructure as predictors, the model estimates clean water accessibility across countries and years. The project combines multiple global datasets, performs preprocessing, handles missing values using K-Nearest Neighbors imputation, and evaluates Linear Regression and Random Forest models.

A trained best-performing model is saved as a `.pkl` file and integrated into a Streamlit app for interactive predictions.

---

## Problem Statement

### Cohort 06: Clean Water Access Forecaster

**SDG Goal:** SDG 6 – Clean Water and Sanitation  
**Description:** Predict the percentage of a population with access to safely managed drinking water based on national GDP and infrastructure investment. The project merges multiple datasets and handles missing values using K-Nearest Neighbors imputation.

---

## Features

- Data sourcing from three international datasets  
- Data cleaning and merging pipeline  
- Conversion of wide-format datasets into a consistent long format  
- Exploratory Data Analysis (EDA)  
- Machine learning model development using scikit-learn  
- Comparison of Linear Regression vs Random Forest  
- Model evaluation using RMSE, MAE, and R²  
- Saving best model as `best_model.pkl`  
- Streamlit application for predictions  

---

## Data Sources

### 1. Safely Managed Drinking Water  
Source: WHO/UNICEF JMP (SDG 6.1.1)

### 2. GDP per Capita (Current US$)  
Source: World Bank

### 3. Access to Electricity (% of Population)  
Source: World Bank

---

## Final Variables

### Target  
- `safe_water_access_pct`

### Features  
- `gdp_per_capita`  
- `access_to_electricity_pct`

---

## Methodology

### Data Preparation

- Standardized dataset formatting  
- Reshaped World Bank datasets from wide to long form  
- Merged datasets using Country Code and Year  
- Removed missing target values  
- Imputed missing features using `KNNImputer`
<img width="644" height="532" alt="image" src="https://github.com/user-attachments/assets/5982eb77-97f5-486c-8463-1af02587412b" />


### Modeling

- Train-test split: 80/20  
- Pipeline: Imputation → Scaling → Model Training  
- Models used:  
  - Linear Regression  
  - Random Forest Regressor (300 trees)

### Evaluation Metrics

| Model           | RMSE   | MAE    | R²    |
|----------------|--------|--------|-------|
| LinearRegression | 17.780 | 13.983 | 0.660 |
| RandomForest     | 14.781 | 9.771  | 0.765 |

### Model Saving

Best model stored at:

```

models/best_model.pkl

```

---

## Project Structure

```

AquaWise/
│── data/
│── models/
│     └── best_model.pkl
│── notebook/
│     └── clean_water_forecaster.ipynb
│── app.py
│── train_model.py
│── requirements.txt
└── README.md

```

---

## Streamlit Application

The Streamlit app allows users to select:

- Country  
- Year  

The model retrieves GDP and electricity data and predicts clean water access.

### Outputs include:

- GDP per capita  
- Electricity access percentage  
- Predicted safe drinking water access percentage  

### Run the App

```

streamlit run app.py

```

---

## How to Run Locally

### Clone Repository

```

git clone <repository-url>
cd AquaWise

```

### Install Dependencies

```

pip install -r requirements.txt

```

### Run Notebook

Open:

```

notebook/clean_water_forecaster.ipynb

```

### Run Streamlit App

```

streamlit run app.py

```

---

## Results

- Both GDP and electricity access show strong positive correlation with clean water access  
- Random Forest performed better than Linear Regression  
- Data imputation significantly improves model stability  
- Economic and infrastructure indicators play a major role in predicting water access  

---

## Future Enhancements

- Add more socioeconomic indicators  
- Hyperparameter tuning  
- Region-specific models  
- SHAP-based model explanation  
- Additional charts in the Streamlit UI  

```



