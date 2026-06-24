
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

# Scale the input data
scaled_input = scaler.transform(input_df)

# Make prediction
if st.button('Predict Water Safety'):
    prediction = model.predict(scaled_input)
    prediction_proba = model.predict_proba(scaled_input)

    st.subheader('Prediction Result:')
    if prediction[0] == 1:
        st.success('The water is predicted to be **SAFE** for consumption.')
    else:
        st.error('The water is predicted to be **UNSAFE** for consumption.')
    
    st.write(f"Confidence (Safe): {prediction_proba[0][1]:.2f}")
    st.write(f"Confidence (Unsafe): {prediction_proba[0][0]:.2f}")

st.sidebar.markdown('---')
st.sidebar.markdown('Developed by Your Name/Organization')
