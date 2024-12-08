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

#%% global variables
    
#custom CSS for main_page
main_page_css = """
    <style>
    .block-container {
        width: 95%; 
        max-width: 900px; 
        display: flex;
        flex-direction: column; 
    }
    .content-box {
        padding-left: 2px; 
        margin-bottom: 0px;
        border-radius: 5px;
        background-color: #E8E8E8;
        margin-bottom: 2px;
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
    .section-header {
        writing-mode: vertical-rl;
        transform: rotate(180deg);
        font-size: 24px;
        margin-right: 20px;
        margin-top: 25%;
        margin-bottom: 25%;  
    }
    /* Style for the expander header when collapsed */
    [data-testid="stExpander"] summary {
        font-size: 30px;
        color: black;
        font-weight: bold;
    }
    /* Style for the expander header when expanded */
    [data-testid="stExpander"] details[open] > summary {
        font-size: 24px;
        color: green;
        font-weight: bold;
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
        font-size: 18px; /* Font size for better visibility */
        cursor: pointer; /* Change cursor to pointer for clickable effect */
        width: 100%; /* Make button take full width of sidebar */
        text-align: left; /* Align text to the left */
        padding: 10px 0; /* Add some padding */
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

    
    /* Target only sidebar button */
    section[data-testid="stSidebar"] div.stButton > button[data-testid="stBaseButton-secondary"] {
        background-color: transparent; /* Make button background invisible */
        border: none; /* Remove border */
        color: black; /* Button text color black */
        font-size: 18px; /* Font size for better visibility */
        cursor: pointer; /* Change cursor to pointer for clickable effect */
        width: 100%; /* Make button take full width of sidebar */
        text-align: left; /* Align text to the left */
        padding: 10px 0; /* Add some padding */
    }
    
    /* Target hover effect for sidebar button */
    section[data-testid="stSidebar"] div.stButton > button[data-testid="stBaseButton-secondary"]:hover {
        background-color: #ccc; /* Grey background on hover */
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
   
#%%% Set the page configuration
def get_classification_with_url():
    paper_id = st.session_state["paper_id"]
    if paper_id:
        try:
            client = st.session_state["client"]
            paper = client.get_paper_reviews(paper_id)
            sentences_json = paper.sentences.to_dict(orient="records")
        except Exception as e:
            st.error(e)
    else:
        st.error("Invalid OpenReview URL.")
    
    response = requests.post(
        "http://localhost:8000/roots_themes",
        json={"data": sentences_json}
    )
    if response.status_code == 200:
        attitude_roots = response.json()
        attitude_roots = pd.DataFrame(attitude_roots)
        # st.session_state.roots_themes = response.json()
        # st.session_state.page = "Meta Reviewer Dashboard"  # Display the classification result
        # st.rerun()
    else:
        st.error(f"Error: {response.text}")
    return attitude_roots

def main_page(custom_css):
    base_path = os.getcwd()
    # Apply custom CSS Styles
    #st.markdown(custom_css, unsafe_allow_html=True)
    st.markdown(main_page_css, unsafe_allow_html=True)
    
    
    st.title("Paper Review Summary")

    attitude_roots = get_classification_with_url()
    
    with open(os.path.join(base_path, 'frontend/dummy_data', 'dummy_overview.pkl'), 'rb') as file:
        overview = pickle.load(file)
    with open(os.path.join(base_path, 'frontend/dummy_data', 'dummy_requests.pkl'), 'rb') as file:
        request_information = pickle.load(file)

    summary = pd.read_csv(os.path.join(base_path,"frontend/dummy_data", "dummy_summary.csv"), sep=";", encoding="utf-8")
    
    use_default_container(modules.overview.show_overview, overview)
    attitude_root_container = lambda: modules.attitude_roots.show_attitude_roots_data(attitude_roots)
    request_information_container = lambda: modules.request_information.show_request_information_data(request_information)
    summary_container = lambda: modules.summary.show_summary_data(summary)
    
    slideshow = ss.StreamlitSlideshow([attitude_root_container, request_information_container, summary_container], ["Attitude Roots", "Request Information", "Summary"])
    use_default_container(slideshow.show)
        
        
            
    use_default_container(modules.contact_info.show_contact_info)