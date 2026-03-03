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
