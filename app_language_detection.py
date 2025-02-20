import streamlit as st
import joblib
import numpy as np

# Load the saved Naïve Bayes model and vectorizer
model = joblib.load("naive_bayes_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Define the Streamlit UI
st.title("🔤 Language Detection App")
st.write("Enter a text snippet, and the model will predict the language.")

# Input field
user_input = st.text_area("Enter text here:", "")

if st.button("Detect Language"):
    if user_input.strip():
        # Transform input text using the vectorizer
        input_features = vectorizer.transform([user_input])

        # Predict language
        prediction = model.predict(input_features)[0]

        # Display result
        st.success(f"Predicted Language: **{prediction}**")
    else:
        st.warning("Please enter some text to predict.")

# Run this using `streamlit run app.py`
