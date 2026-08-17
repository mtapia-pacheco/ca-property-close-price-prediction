from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


CURRENT_YEAR = 2026

MODEL_PATHS = [
    Path("./final_model_outputs/final_model.joblib"),
]

DEFAULT_VALUES = {
    "GarageSpaces": 0,
    "ParkingTotal": 0,
    "AssociationFee": 0,
    "LotSizeAcres_missing": 0,
    "GarageSpaces_missing": 1,
    "AssociationFee_missing": 1,
    "City": "Unknown",
    "CountyOrParish": "Unknown",
    "PostalCode": "Unknown",
    "MLSAreaMajor": "Unknown",
    "HighSchoolDistrict": "Unknown",
    "SchoolDistrict": "Unknown",
}


st.set_page_config(
    page_title="California Property Price Predictor",
    page_icon="🏠",
    layout="centered",
)


@st.cache_resource
def load_model():
    for model_path in MODEL_PATHS:
        if model_path.exists():
            return joblib.load(model_path)

    paths_checked = ", ".join(str(path) for path in MODEL_PATHS)
    raise FileNotFoundError(f"Model file not found. Checked: {paths_checked}")


try:
    model = load_model()
except Exception as error:
    st.error("The trained model could not be loaded.")
    st.exception(error)
    st.stop()


st.title("California Property Price Predictor")
st.write("Enter basic property information to estimate the final close price.")

st.subheader("Property Details")

living_area = st.number_input(
    "Living Area (sq ft)",
    min_value=200,
    max_value=20000,
    value=1800,
    step=50,
)

beds = st.number_input(
    "Bedrooms",
    min_value=1,
    max_value=20,
    value=3,
    step=1,
)

baths = st.number_input(
    "Bathrooms",
    min_value=1,
    max_value=20,
    value=2,
    step=1,
)

lot_size = st.number_input(
    "Lot Size (acres)",
    min_value=0.01,
    max_value=20.0,
    value=0.15,
    step=0.01,
)

year_built = st.number_input(
    "Year Built",
    min_value=1800,
    max_value=CURRENT_YEAR,
    value=1980,
    step=1,
)


if st.button("Predict Price"):
    input_data = pd.DataFrame(
        {
            "LivingArea": [living_area],
            "BedroomsTotal": [beds],
            "BathroomsTotalInteger": [baths],
            "LotSizeAcres": [lot_size],
            "YearBuilt": [year_built],
        }
    )

    input_data["PropertyAge"] = CURRENT_YEAR - input_data["YearBuilt"]
    input_data["BedBathRatio"] = (
        input_data["BedroomsTotal"] / input_data["BathroomsTotalInteger"]
    )
    input_data["LivingAreaPerBedroom"] = (
        input_data["LivingArea"] / input_data["BedroomsTotal"]
    )

    # These defaults keep this first app version simple
    for col, value in DEFAULT_VALUES.items():
        if col not in input_data.columns:
            input_data[col] = value

    expected_features = getattr(model, "feature_names_in_", None)
    if expected_features is not None:
        for col in expected_features:
            if col not in input_data.columns:
                input_data[col] = DEFAULT_VALUES.get(col, 0)
        input_data = input_data[list(expected_features)]

    try:
        prediction = model.predict(input_data)[0]

        st.subheader("Predicted Close Price")
        st.success(f"${prediction:,.0f}")

        st.write("Input used for prediction:")
        st.dataframe(input_data, use_container_width=True)
    except Exception as error:
        st.error("Prediction failed. Check that the app input columns match the model features.")
        st.exception(error)

# Run locally with:
# streamlit run app.py
