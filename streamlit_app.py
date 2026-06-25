
import streamlit as st
import pandas as pd
import joblib

# Load the saved model, scaler, and feature ranges
model = joblib.load('xgboost_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_ranges = joblib.load('feature_ranges.pkl') # Load feature ranges

# Get the feature names in the order they were used for training
original_feature_names = scaler.feature_names_in_

# Define the top 5 most important features based on the previous analysis
top_5_features = ['cadmium', 'aluminium', 'silver', 'viruses', 'uranium']

# Title of the app
st.title('Water Quality Prediction App (Top Features)')
st.write('Adjust the parameters for the top 5 most important features to predict water safety.')

# Create input fields for the top 5 features using sliders
input_data = {}
for feature in top_5_features:
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

# For the features not in top_5_features, use their mean values
full_input_data = input_data.copy()
for feature in original_feature_names:
    if feature not in full_input_data:
        full_input_data[feature] = feature_ranges[feature]['mean']

# Ensure the order of features matches the training data
input_df = pd.DataFrame([full_input_data], columns=original_feature_names)

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
