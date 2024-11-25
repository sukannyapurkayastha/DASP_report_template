import streamlit as st
from landing_page import landing_page
from main_page import main_page

#%% Define necessary global variables
# Define custom CSS styles

custom_css = """
    <style>
    body {
        background-color: #F2F2F2 !Important;
    }
    .stApp {
        background-color: #F2F2F2; /* For Streamlit app container */
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
    </style>
    """


#%% Implement Pagelogic

# Initialize session state
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "landing_page"

# Page routing
if st.session_state["current_page"] == "landing_page":
    landing_page(custom_css)
    
elif st.session_state["current_page"] == "main_page":
    main_page(custom_css)

