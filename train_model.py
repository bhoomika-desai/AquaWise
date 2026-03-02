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

