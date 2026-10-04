import streamlit as st
import pandas as pd
import joblib


# Sidebar
st.sidebar.title("Insurance Predictor")

st.sidebar.write(
    "Enter customer details to estimate the insurance charge."
)

st.sidebar.markdown("---")

st.sidebar.write(
    "Model: Random Forest Regression"
)

st.sidebar.write(
    "Built with Python and Streamlit"
)

if st.sidebar.button("Reset Inputs"):
    st.rerun()


# Main title
st.title("Insurance Charge Prediction")

st.write(
    "This app predicts medical insurance charges using a trained "
    "Random Forest Regression model."
)


# Load trained model and scaler
model = joblib.load("Insurance_Random_Forest_Tuned.pkl")
scaler = joblib.load("insurance_scaler.pkl")


# Customer details section
st.subheader("Enter Customer Details")

st.write(
    "Provide the details below to estimate the insurance charge."
)

st.markdown("### Customer Information")


# Age and Sex
col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=1,
        max_value=100,
        value=30
    )

with col2:
    is_female = st.selectbox(
        "Sex",
        ["Male", "Female"]
    )


# BMI and Children
col3, col4 = st.columns(2)

with col3:
    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=60.0,
        value=25.0
    )

with col4:
    children = st.number_input(
        "Number of Children",
        min_value=0,
        max_value=10,
        value=0
    )


# Smoker and Region
col5, col6 = st.columns(2)

with col5:
    is_smoker = st.selectbox(
        "Smoker",
        ["No", "Yes"]
    )

with col6:
    region = st.selectbox(
        "Region",
        ["northeast", "northwest", "southeast", "southwest"]
    )


# Divider
st.divider()


# Prediction button
if st.button(
    "Predict Insurance Charge",
    type="primary"
):

    # Convert categorical values to 0/1
    is_female_value = 1 if is_female == "Female" else 0
    is_smoker_value = 1 if is_smoker == "Yes" else 0


    # Create BMI category
    if bmi <= 18.5:
        bmi_category = "UnderWeight"

    elif bmi <= 24.9:
        bmi_category = "Normal"

    elif bmi <= 29.9:
        bmi_category = "OverWeight"

    else:
        bmi_category = "Obese"


    # Create input dataframe
    input_data = pd.DataFrame({
        "age": [age],
        "isFemale": [is_female_value],
        "bmi": [bmi],
        "children": [children],
        "isSmoker": [is_smoker_value],

        "region_northwest": [
            1 if region == "northwest" else 0
        ],

        "region_southeast": [
            1 if region == "southeast" else 0
        ],

        "region_southwest": [
            1 if region == "southwest" else 0
        ],

        "bmi_category_Normal": [
            1 if bmi_category == "Normal" else 0
        ],

        "bmi_category_OverWeight": [
            1 if bmi_category == "OverWeight" else 0
        ],

        "bmi_category_Obese": [
            1 if bmi_category == "Obese" else 0
        ]
    })


    # Scale numerical features
    input_data[["age", "bmi", "children"]] = scaler.transform(
        input_data[["age", "bmi", "children"]]
    )


    # Make prediction
    prediction = model.predict(input_data)


    # Display result
    st.subheader("Prediction Result")

    st.success(
        "Prediction generated successfully!"
    )

    st.write(
        f"**BMI Category:** {bmi_category}"
    )

    st.metric(
        "Predicted Insurance Charge",
        f"₹{prediction[0]:,.2f}"
    )

    st.info(
        "This prediction is generated using a trained "
        "Random Forest regression model."
    )


    # Prediction range message
    if prediction[0] < 10000:
        st.info(
            "The predicted insurance charge is relatively low."
        )

    elif prediction[0] < 30000:
        st.warning(
            "The predicted insurance charge is moderate."
        )

    else:
        st.error(
            "The predicted insurance charge is relatively high."
        )


# Footer
st.markdown("---")

st.caption(
    "Insurance Charge Prediction App | Built with Python, "
    "Scikit-learn and Streamlit"
)