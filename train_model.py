from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent


def load_world_bank_indicator(csv_path: Path, value_name: str) -> pd.DataFrame:
    """
    Load a World Bank wide-format indicator CSV (like GDP or access to electricity)
    and return it in long / tidy format:
    columns: Country Name, Country Code, Year, <value_name>
    """
    df = pd.read_csv(csv_path, skiprows=4)

    # Keep only the useful ID columns and year columns
    id_cols = ["Country Name", "Country Code", "Indicator Name", "Indicator Code"]
    year_cols = [c for c in df.columns if c not in id_cols]

    long_df = df.melt(
        id_vars=["Country Name", "Country Code"],
        value_vars=year_cols,
        var_name="Year",
        value_name=value_name,
    )

    long_df["Year"] = pd.to_numeric(long_df["Year"], errors="coerce")
    long_df[value_name] = pd.to_numeric(long_df[value_name], errors="coerce")

    # Drop rows without a valid year
    long_df = long_df.dropna(subset=["Year"])
    long_df["Year"] = long_df["Year"].astype(int)

    return long_df


def load_water_access(csv_path: Path) -> pd.DataFrame:
    """
    Load the safely managed drinking water dataset from Our World in Data.
    Expected columns: Entity, Code, Year, Share of the population using safely managed drinking water
    """
    df = pd.read_csv(csv_path)

    df = df.rename(
        columns={
            "Entity": "Country Name",
            "Code": "Country Code",
            "Year": "Year",
            "Share of the population using safely managed drinking water": "safe_water_access_pct",
        }
    )

    # Keep only rows that have an ISO-style country code (3 letters) to avoid aggregates
    df = df[df["Country Code"].str.len() == 3]

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype(int)
    df["safe_water_access_pct"] = pd.to_numeric(
        df["safe_water_access_pct"], errors="coerce"
    )

    return df[["Country Name", "Country Code", "Year", "safe_water_access_pct"]]

def build_dataset() -> pd.DataFrame:
    """Merge GDP per capita, access to electricity, and safe water access."""
    gdp_path = BASE_DIR / "API_NY.GDP.PCAP.CD_DS2_en_csv_v2_46.csv"
    elec_path = BASE_DIR / "API_EG.ELC.ACCS.ZS_DS2_en_csv_v2_158.csv"
    water_path = BASE_DIR / "proportion-using-safely-managed-drinking-water.csv"

    gdp = load_world_bank_indicator(gdp_path, "gdp_per_capita")
    elec = load_world_bank_indicator(elec_path, "access_to_electricity_pct")
    water = load_water_access(water_path)

    # Merge step by step on Country Code and Year
    merged = water.merge(
        gdp[["Country Code", "Year", "gdp_per_capita"]],
        on=["Country Code", "Year"],
        how="inner",
    )
    merged = merged.merge(
        elec[["Country Code", "Year", "access_to_electricity_pct"]],
        on=["Country Code", "Year"],
        how="inner",
    )

    # Basic cleaning: drop rows where target is missing
    merged = merged.dropna(subset=["safe_water_access_pct"])

    return merged
    def train_and_evaluate():
    df = build_dataset()

    feature_cols = ["gdp_per_capita", "access_to_electricity_pct"]
    target_col = "safe_water_access_pct"

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "LinearRegression": make_pipeline(LinearRegression()),
        "RandomForest": make_pipeline(
            RandomForestRegressor(n_estimators=300, random_state=42)
        ),
    }
    results = {}
    best_model_name = None
    best_model = None
    best_r2 = -np.inf

    print("Number of samples:", len(df))
    print("Training samples:", len(X_train), "Test samples:", len(X_test))

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results[name] = {"RMSE": rmse, "MAE": mae, "R2": r2}

        print(f"\n{name}")
        print(f"  RMSE: {rmse:.3f}")
        print(f"  MAE : {mae:.3f}")
        print(f"  R2  : {r2:.3f}")

        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_model = model

    print(f"\nBest model: {best_model_name} (R2 = {best_r2:.3f})")
    
# Save best model
    models_dir = BASE_DIR / "models"
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / "best_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": best_model, "feature_cols": feature_cols}, f)

    print(f"Saved best model to: {model_path}")


if __name__ == "__main__":
    train_and_evaluate()


