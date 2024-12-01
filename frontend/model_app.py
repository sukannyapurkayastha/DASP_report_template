import streamlit as st
import requests

# FastAPI endpoint URL
API_URL = "http://127.0.0.1:8000/predict/"

# Set the title of the app
st.title("Model Prediction")

# Create a text input field for the user to input a sentence
sentence = st.text_input("Enter a sentence to predict the description:")

# Display the prediction when the button is clicked
if sentence:
    # Prepare the payload with the input sentence
    payload = {"text": sentence}
    
    # Make the POST request to the FastAPI endpoint
    response = requests.post(API_URL, json=payload)
    
    # Handle the response
    if response.status_code == 200:
        result = response.json()
        st.write("Summary Result:")
        st.write(f"Original Text: {result['text']}")
        st.write(f"Summary: {result['summary']}")
    else:
        st.write(f"Error: {response.status_code}, Could not retrieve prediction.")
