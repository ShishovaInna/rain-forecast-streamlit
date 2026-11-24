import streamlit as st
import pandas as pd
import numpy as np
import joblib

@st.cache_resource
def load_artifact(file_path: str):
    return joblib.load(file_path)

artifact = load_artifact("rain_forest_artifact.joblib")

model = artifact["model"]
imputer = artifact["imputer"]
scaler = artifact["scaler"]
encoder = artifact["encoder"]
input_cols = artifact["input_cols"]
numeric_cols = artifact["numeric_cols"]
categorical_cols = artifact["categorical_cols"]

# Заголовок застосунку
st.title('Australia Rain Tomorrow Predictor')
st.write(
    "Enter today's weather conditions below to predict the chance of rain tomorrow. "
    "The prediction is made by a Random Forest model trained on historical weather data from Australia."
)
    
# Введення даних
col1, col2 = st.columns(2)

with col1:
    min_temp = st.slider("MinTemp", min_value=-8.5, max_value=33.0, value=12.0, step=0.5)
    humidity = st.slider("Humidity", min_value=0.0, max_value=100.0, value=68.0, step=1.0)
    rain_today = st.selectbox("RainToday", ["No", "Yes"])

with col2:
    max_temp = st.slider("MaxTemp", min_value=-4.0, max_value=48.0, value=12.0, step=0.5)
    pressure = st.slider("Pressure", min_value=980.0, max_value=1045.0, value=1017.0, step=0.5)

    if "Location" in categorical_cols:
        loc_idx = categorical_cols.index("Location")
        location_options = list(encoder.categories_[loc_idx])
        location = st.selectbox("Location", location_options)
    else:
        location = "Albury" 

row_data = {}

for col in input_cols:
    if col in numeric_cols:
        row_data[col] = np.nan
    elif col in categorical_cols:
        cat_idx = categorical_cols.index(col)
        row_data[col] = encoder.categories_[cat_idx][0]

row_data["MinTemp"] = min_temp
row_data["MaxTemp"] = max_temp
row_data["Humidity3pm"] = humidity
row_data["Pressure3pm"] = pressure
row_data["RainToday"] = rain_today
row_data["Location"] = location

input_df = pd.DataFrame([row_data])[input_cols]

def predict_rain(df: pd.DataFrame, model, imputer, scaler, encoder, num_cols: list, cat_cols: list):
    num_processed = imputer.transform(df[num_cols])
    num_scaled = scaler.transform(num_processed)
    cat_encoded = encoder.transform(df[cat_cols])

    if hasattr(cat_encoded, "toarray"):
        cat_encoded = cat_encoded.toarray()

    x_ready = np.hstack([num_scaled, cat_encoded])
    prediction = model.predict(x_ready)[0]
    probabilities = model.predict_proba(x_ready)[0]

    prob_dict = {
        cls: round(float(prob), 2)
        for cls, prob in zip(model.classes_, probabilities)
    }
    return prediction, prob_dict
    
if st.button("Predict"):
    pred, probs = predict_rain(
        input_df, model, imputer, scaler, encoder, numeric_cols, categorical_cols
    )

    prob_no = probs.get("No", 0.0)
    prob_yes = probs.get("Yes", 0.0)

    with st.container(border=True):
            if pred == "Yes":
                st.markdown("### Prediction: :red[Yes] — rain is likely tomorrow")
            else:
                st.markdown("### Prediction: :blue[No] — rain is unlikely tomorrow")

    m1, m2 = st.columns(2)
    m1.metric("Probability of No", f"{prob_no * 100:.0f}%")
    m2.metric("Probability of Yes", f"{prob_yes * 100:.0f}%")

    st.write("Rain probability bar:")
    st.progress(min(max(float(prob_yes), 0.0), 1.0))