from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Clean Water Access Forecaster",
    page_icon="💧",
    layout="centered",
)

st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(135deg, #e0f7fa 0%, #f1f8e9 100%);
    }
    .stMetric {
        background-color: white;
        border-radius: 0.75rem;
        padding: 0.75rem 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"


def load_world_bank_indicator(csv_path: Path, value_name: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, skiprows=4)
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
    long_df = long_df.dropna(subset=["Year"])
    long_df["Year"] = long_df["Year"].astype(int)
    return long_df


def load_water_access(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.rename(
        columns={
            "Entity": "Country Name",
            "Code": "Country Code",
            "Year": "Year",
            "Share of the population using safely managed drinking water": "safe_water_access_pct",
        }
    )
    df = df[df["Country Code"].str.len() == 3]
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype(int)
    df["safe_water_access_pct"] = pd.to_numeric(
        df["safe_water_access_pct"], errors="coerce"
    )
    return df[["Country Name", "Country Code", "Year", "safe_water_access_pct"]]


@st.cache_resource
def load_model():
    import pickle

    with open(MODEL_PATH, "rb") as f:
        saved = pickle.load(f)
    return saved["model"], saved["feature_cols"]


@st.cache_resource
def load_dataset() -> pd.DataFrame:
    gdp_path = BASE_DIR / "API_NY.GDP.PCAP.CD_DS2_en_csv_v2_46.csv"
    elec_path = BASE_DIR / "API_EG.ELC.ACCS.ZS_DS2_en_csv_v2_158.csv"
    water_path = BASE_DIR / "proportion-using-safely-managed-drinking-water.csv"

    gdp = load_world_bank_indicator(gdp_path, "gdp_per_capita")
    elec = load_world_bank_indicator(elec_path, "access_to_electricity_pct")
    water = load_water_access(water_path)

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
    # Keep only rows where both features are present so the app never shows NaN
    merged = merged.dropna(subset=["gdp_per_capita", "access_to_electricity_pct"])
    return merged


st.title("💧 Clean Water Access Forecaster")
st.caption("SDG 6 – Clean Water and Sanitation")

st.write(
    "Explore how **economic development** and **infrastructure** relate to access to "
    "safely managed drinking water around the world."
)

try:
    model, feature_cols = load_model()
except FileNotFoundError:
    st.error(
        "Model file not found. Please run your notebook to train and save "
        "the best model to `models/best_model.pkl`."
    )
    st.stop()

df = load_dataset()

with st.sidebar:
    st.header("Scenario selection")
    country_name = st.selectbox(
        "Country",
        sorted(df["Country Name"].unique()),
    )

    country_df = df[df["Country Name"] == country_name]
    years = sorted(country_df["Year"].unique())
    year = st.selectbox("Year", years, index=len(years) - 1)

    run_button = st.button("Predict water access")

row = df[(df["Country Name"] == country_name) & (df["Year"] == year)].iloc[0]
gdp_per_capita = float(row["gdp_per_capita"])
access_to_electricity_pct = float(row["access_to_electricity_pct"])

col1, col2 = st.columns(2)
with col1:
    st.metric("GDP per capita (USD)", f"{gdp_per_capita:,.0f}")
with col2:
    st.metric("Access to electricity", f"{access_to_electricity_pct:.1f}%")

st.divider()

if run_button:
    input_df = pd.DataFrame(
        [[gdp_per_capita, access_to_electricity_pct]], columns=feature_cols
    )
    prediction = float(model.predict(input_df)[0])

    col_pred, col_chart = st.columns([1, 1])

    with col_pred:
        st.subheader("Predicted water access")
        st.metric(
            "Safely managed drinking water",
            f"{prediction:.2f}%",
        )

        st.markdown(
            f"For **{country_name}** in **{year}**, with GDP per capita of "
            f"**{gdp_per_capita:,.0f} USD** and **{access_to_electricity_pct:.1f}%** "
            f"access to electricity, the model estimates that **{prediction:.2f}%** "
            f"of the population uses safely managed drinking water."
        )

    with col_chart:
        chart_df = pd.DataFrame(
            {
                "Indicator": [
                    "Access to electricity (%)",
                    "Predicted safe water access (%)",
                ],
                "Value": [
                    access_to_electricity_pct,
                    prediction,
                ],
            }
        )
        st.bar_chart(chart_df.set_index("Indicator"))
