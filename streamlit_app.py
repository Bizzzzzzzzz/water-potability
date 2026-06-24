
import streamlit as st
import pandas as pd
import joblib

# Load the saved model and scaler
model = joblib.load('xgboost_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_ranges = joblib.load('feature_ranges.pkl') # Load feature ranges

# Title of the app
st.title('Water Quality Prediction App')
st.write('Enter the water quality parameters to predict if the water is safe for consumption.')

# Define all critical thresholds
DANGEROUS_THRESHOLDS = {
    'aluminium': 2.8,
    'ammonia': 32.5,
    'arsenic': 0.01,
    'barium': 2.0,
    'cadmium': 0.005,
    'chloramine': 4.0,
    'chromium': 0.1,
    'copper': 1.3,
    'flouride': 1.5,
    'bacteria': 0.0,
    'viruses': 0.0,
    'lead': 0.015,
    'nitrates': 10.0,
    'nitrites': 1.0,
    'mercury': 0.002,
    'perchlorate': 56.0,
    'radium': 5.0,
    'selenium': 0.5,
    'silver': 0.1,
    'uranium': 0.3
}

# Create input fields for all features using sliders
feature_names = scaler.feature_names_in_.tolist()

input_data = {}
for feature in feature_names:
    min_val = feature_ranges[feature]['min']
    max_val = feature_ranges[feature]['max']
    mean_val = feature_ranges[feature]['mean']
    
    # Use st.sidebar.slider for each input
    input_data[feature] = st.sidebar.slider(
        f'Enter {feature}',
        min_value=float(min_val),
        max_value=float(max_val),
        value=float(mean_val) # Default to mean value
    )

# Convert input data to a DataFrame
input_df = pd.DataFrame([input_data])

# Make prediction
if st.button('Predict Water Safety'):
    st.subheader('Prediction Result:')
    
    is_unsafe_due_to_threshold = False
    for feature, threshold in DANGEROUS_THRESHOLDS.items():
        if input_data[feature] > threshold:
            st.error(f'The water is predicted to be **UNSAFE** for consumption due to high {feature.capitalize()} levels (>{threshold}).')
            st.write('This is based on an immediate safety rule, overriding the machine learning model.')
            is_unsafe_due_to_threshold = True
            break # Stop checking once one dangerous threshold is met

    if not is_unsafe_due_to_threshold:
        # Scale the input data for the ML model
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)
        prediction_proba = model.predict_proba(scaled_input)

        if prediction[0] == 1:
            st.success('The water is predicted to be **SAFE** for consumption.')
        else:
            st.error('The water is predicted to be **UNSAFE** for consumption.')
        
        st.write(f"Confidence (Safe): {prediction_proba[0][1]:.2f}")
        st.write(f"Confidence (Unsafe): {prediction_proba[0][0]:.2f}")

st.sidebar.markdown('---')
st.sidebar.markdown('Developed by Your Name/Organization')
