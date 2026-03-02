# Clean Water Access Forecaster (Simple Version)

This project predicts the percentage of a population using safely managed drinking water (SDG 6.1.1) based on:

- GDP per capita (current US$, World Bank)
- Access to electricity (% of population, World Bank)

## Data files

Place the following CSV files in the project root (already done in this template):

- `API_NY.GDP.PCAP.CD_DS2_en_csv_v2_46.csv` – GDP per capita
- `API_EG.ELC.ACCS.ZS_DS2_en_csv_v2_158.csv` – Access to electricity
- `proportion-using-safely-managed-drinking-water.csv` – Safely managed drinking water (Our World in Data)

## Environment setup

```bash
pip install -r requirements.txt
```

## Train and evaluate models

```bash
python train_model.py
```

This script:

- Cleans and merges the three datasets
- Splits into train/test sets
- Trains **Linear Regression** and **Random Forest** models
- Evaluates using **RMSE**, **MAE**, and **R²**
- Saves the best model to `models/best_model.pkl`

## Run the Streamlit app

```bash
streamlit run app.py
```

The app lets you input GDP per capita and access to electricity and outputs the predicted share of population with access to safely managed drinking water.

