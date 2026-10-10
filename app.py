 
import streamlit as st
import pandas as pd
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏡",
    layout="centered"
)

# Load model and dataset
MODEL_PATH = "models/best_house_price_model.joblib"
DATA_PATH = "data/processed/house_price_cleaned.csv"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

if not os.path.exists(MODEL_PATH) or not os.path.exists(DATA_PATH):
    st.error("Model or dataset file not found. Please run train_model.py first.")
    st.stop()

model = load_model()
df = load_data()

st.title("🏡 House Price Prediction")
st.write(
    "Enter the property details below to estimate the house price "
    "using a machine learning model."
)

st.info("This app provides an estimate, not a guaranteed market price.")

# Input form
with st.form("house_prediction_form"):
    st.subheader("Property Details")

    area = st.number_input(
        "Area (square feet)",
        min_value=100.0,
        max_value=20000.0,
        value=float(df["Area_SqFt"].median()),
        step=50.0
    )

    rooms = st.number_input(
        "Number of Rooms",
        min_value=1,
        max_value=20,
        value=int(df["Rooms"].median()),
        step=1
    )

    build_year = st.number_input(
        "Year Built",
        min_value=1800,
        max_value=2026,
        value=int(df["Build_Year"].median()),
        step=1
    )

    location = st.selectbox(
        "Location",
        sorted(df["Location"].dropna().unique().tolist())
    )

    street_type = st.selectbox(
        "Street Type",
        sorted(df["Street_Type"].dropna().unique().tolist())
    )

    furnishing = st.selectbox(
        "Furnishing",
        sorted(df["Furnishing"].dropna().unique().tolist())
    )

    property_type = st.selectbox(
        "Property Type",
        sorted(df["Property_Type"].dropna().unique().tolist())
    )

    has_pool = st.selectbox(
        "Has Pool?",
        sorted(df["Has_Pool"].dropna().unique().tolist())
    )

    submitted = st.form_submit_button("Predict House Price")

if submitted:
    input_data = pd.DataFrame([{
        "Area_SqFt": area,
        "Rooms": rooms,
        "Build_Year": build_year,
        "Location": location,
        "Street_Type": street_type,
        "Furnishing": furnishing,
        "Property_Type": property_type,
        "Has_Pool": has_pool
    }])

    try:
        predicted_price = model.predict(input_data)[0]

        st.success("Prediction completed!")
        st.metric(
            label="Estimated House Price",
            value=f"{predicted_price:,.2f}"
        )

        st.caption(
            "This is a machine-learning estimate based on your dataset. "
            "Actual property prices may differ."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
