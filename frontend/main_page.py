import streamlit as st
from streamlit_extras.stylable_container import stylable_container
import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pickle
import modules.overview
import modules.attitude_roots
import modules.request_information
import modules.summary
import modules.contact_info
import modules.slideshow as ss
from modules.shared_methods import use_default_container
import requests
import sys

# Fügen Sie den übergeordneten Pfad hinzu
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Jetzt können Sie importieren

# from backend.model_backend import classify_paper

# %% global variables

# custom CSS for main_page
main_page_css = """
    <style>
    h1, h2, h3, p, div, span {
        max-width: 100%;
        word-wrap: break-word;
        overflow-wrap: break-word;
    }
    .block-container {
        width: 100%; 
        max-width: 920px; 
        display: flex;
        flex-direction: column; 
    }
    .invisbible-line-big {
        height: 60px;
    }
    .invisbible-line-small {
        height: 30px;
    }
    .invisbible-line-minor {
        height: 5px;
    }
    /* Hide the icon next to headers */
    [data-testid="stHeaderActionElements"] {
        display: none !important;
    }
    /* Hide the fullscreen button next to pie charts */
    [data-testid="StyledFullScreenButton"] {
        display: none !important;
    }
    /* Target only regular button */
    .stButton button {
        background-color: transparent; /* Make button background invisible */
        border: none; /* Remove border */
        color: black; /* Button text color black */
    }
    /* Target only regular button on hover*/
    .stButton button:hover {
        background-color: #ccc; /* Grey background on hover */
        color: #3a60b2
    }
    
    /* Download Button styling */
    .stDownloadButton button {
        background-color: #3a60b2 !important;
        color: white !important;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
        cursor: pointer;
    }
    .stDownloadButton button:hover {
        background-color: #0056b3 !important;
    }

    .popover-open {
    width: 200px;
    height: 100px;
    position: absolute;
    inset: unset;
    bottom: 5px;
    right: 5px;
    margin: 0;
    }
 
    /* Style the popover container */
    .stPopover {
        width: 100%;
        background-color: transparent; /* Transparent background */
        text-align: center; /* Center the content */
        display: flex;
        justify-content: center;
        align-items: center;
        
    }
    
    /* Style popover button */
    .stPopover button {
        background-color: transparent; /* Transparent background */
        text-align: center; /* Center the content */
        border: none;
    }
    [data-testid="stPopover"] button:active {
        background-color: transparent;
        }
    [data-testid="stPopoverBody"] {
        background-color: white;
    }
    [data-testid="stHorizontalBlock"] {
        background-color: white;
    }
    </style>
    """


# %%% Set the page configuration
def get_classification_with_api():
    try:
        # Überprüfen, ob die benötigten Variablen existieren
        # if "paper_id" not in st.session_state or not st.session_state["paper_id"]:
        #     st.error("Invalid OpenReview URL.")
        #     return pd.DataFrame()  # Rückgabe eines leeren DataFrame

        # if not client:
        #     st.error("Client is not initialized.")
        #     return pd.DataFrame()  # Rückgabe eines leeren DataFrame

        if "reviews" not in st.session_state:
            st.error("No reviews to analyze")

        payload = [review.__dict__ for review in st.session_state.reviews]

        response = requests.post(
            "http://localhost:8080/process",
            json={"data": payload}
        )

        if response.status_code == 200:
            data = response.json()
            # df_sentences = pd.DataFrame(data["df_sentences"])
            df_overview = pd.DataFrame(data["overview"])
            df_requests = pd.DataFrame(data["request_response"])
            df_attitudes = pd.DataFrame(data["attitude_response"])
            df_summary = pd.DataFrame(data["summary_response"])
            # Todo: add other returned data
        else:
            st.error(f"Error: {response.text}")

        # Todo: Return all the dataframes (once we returned them from the api)
        return df_overview, df_requests, df_attitudes, df_summary

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return pd.DataFrame()  # Rückgabe eines leeren DataFrame


def main_page(custom_css):
    base_path = os.getcwd()
    # Apply custom CSS Styles
    st.markdown(main_page_css, unsafe_allow_html=True)

    if "main_page_variables" not in st.session_state:
        # Fetch data and store it in session state
        overview, request_information, attitude_roots, summary = get_classification_with_api()
        st.session_state.main_page_variables = {
            "overview": overview,
            "request_information": request_information,
            "attitude_roots": attitude_roots,
            "summary": summary
        }

    # Access data from session state
    overview = st.session_state.main_page_variables["overview"]
    attitude_roots = st.session_state.main_page_variables["attitude_roots"]
    request_information = st.session_state.main_page_variables["request_information"]
    summary = st.session_state.main_page_variables["summary"]

    if overview.empty:
        st.warning("No data available for overview.")
        with open(os.path.join('dummy_data', 'dummy_overview.pkl'), 'rb') as file:
            overview = pickle.load(file)
    if attitude_roots.empty:
        st.warning("No data available for attitudes classification.")
        with open(os.path.join('dummy_data', 'dummy_attitude_roots.pkl'), 'rb') as file:
            attitude_roots = pickle.load(file)
    if request_information.empty:
        st.warning("No data available for request classification.")
        with open(os.path.join(base_path, 'dummy_data', 'dummy_requests.pkl'), 'rb') as file:
            request_information = pickle.load(file)
    if summary.empty:
        st.warning("No data available for summary - display dummy data instead!") 
        summary = pd.read_csv(os.path.join(base_path, "dummy_data", "dummy_summary.csv"), sep=";", encoding="utf-8")

    # Show overview container
    use_default_container(modules.overview.show_overview, overview)
    
    # Show slideshow containing attitude roots, request information and summary
    attitude_root_container = lambda: modules.attitude_roots.show_attitude_roots_data(attitude_roots)
    request_information_container = lambda: modules.request_information.show_request_information_data(request_information)
    summary_container = lambda: modules.summary.show_summary_data(summary)

    slideshow = ss.StreamlitSlideshow([attitude_root_container, request_information_container, summary_container],
                                      ["Attitude Roots", "Request Information", "Summary"])
    use_default_container(slideshow.show)

    # Show contact info
    use_default_container(modules.contact_info.show_contact_info)
