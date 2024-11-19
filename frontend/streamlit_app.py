import time
import streamlit as st
import pandas as pd
import os

attitude_roots = pd.read_csv(os.path.join("frontend", "dummy_data", "dummy_attitude_roots.csv"), sep=";", encoding="utf-8")
request_information = pd.read_csv(os.path.join("frontend", "dummy_data", "dummy_request_information.csv"), sep=";", encoding="utf-8")
summary = pd.read_csv(os.path.join("frontend", "dummy_data", "dummy_summary.csv"), sep=";", encoding="utf-8")


# Set up the page layout
st.set_page_config(page_title="Review Overview Generation for Papers", layout="wide")

# Page title
st.title("Review Overview Generation for Papers")

# Helper function to check URL validity
def is_valid_url():
    url = st.session_state.get("text_input", "")
    return "https" in url

# Main layout divided into columns
col1, col2, col3, col4 = st.columns([0.5, 1, 3, 0.5])

# First column content
with col2:
    st.header("Previous Requests")
    st.write("No previous requests")

# Second column content
with col3:
    st.header("Overview Generator")

    # Text input and button
    url = st.text_input("Enter the openreview.com URL", key="text_input")
    submit_button = st.button("Submit", key="action_button")

    # Initialize session state variables if not already set
    if "submit_message" not in st.session_state:
        st.session_state.submit_message = ""
    if "show_overview" not in st.session_state:
        st.session_state.show_overview = False
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "show_analyze" not in st.session_state:    
        st.session_state.show_analyze = False

    # Button click logic
    if submit_button:
        if is_valid_url():
            st.session_state.submit_message = "analyzing reviews..."
            st.session_state.show_analyze = True
            st.session_state.processing = True
        else:
            st.session_state.submit_message = "Please enter a valid URL with 'https'"
            st.session_state.processing = False


    # Display the message
    if st.session_state.show_analyze:
        st.write(st.session_state.submit_message)
        st.session_state.show_analyze = False



    if st.session_state.processing:
        time.sleep(1)
        st.session_state.show_overview = True
        st.session_state.processing = False  # Set processing to False to stop updating
        st.write(st.session_state.submit_message)


# Second row that contains results
if st.session_state.show_overview:
    st.markdown("---")
    st.subheader("Overview")
    st.write("Here you can see the results of:")
    st.write("Summary")
    st.write("Attitude root classification")
    st.write("Request Classification")
