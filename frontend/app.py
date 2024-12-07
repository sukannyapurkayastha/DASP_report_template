import streamlit as st
import base64
from pathlib import Path
import main_page
import landing_page
import home_page
import os
import contact
import about

st.set_page_config(
    page_title="Paper Review Generator",
    page_icon=os.path.join("frontend/images" ,"logo.png")
    )

custom_css = """
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
    /* Regular Button styling */
    .stButton button {
        background-color: #3a60b2 !important;
        color: white !important;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
        cursor: pointer;
    }
    .stButton button:hover {
        background-color: #0056b3 !important;
    }
    
    /* Download Button styling (at upload tab) */
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
     /* Style the popover container at overview */
    .popover-open {
    width: 200px;
    height: 100px;
    position: absolute;
    inset: unset;
    bottom: 5px;
    right: 5px;
    margin: 0;
    background-color: white !important;
    }
    .stPopover {
        width: 100%;
        background-color: transparent; /* Transparent background */
        text-align: center; /* Center the content */
        display: flex;
        justify-content: center;
        align-items: center;       
    }   
    /* Style Step boxes on home screen */
    .step-box {
    background-color: #bcd7f2;
    padding: 20px;
    border-radius: 5px;
    margin: 15px auto; /* Center horizontally */
    width: 85%; /* Adjust width as needed */
    box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
}
    </style>
    """

def show_navigation_bar_and_content():

    def load_logo(filepath):
        with open(filepath, "rb") as logo_file:
            return base64.b64encode(logo_file.read()).decode("utf-8")

    # Path to your local logo file
    base_path = Path(__file__).parent
    logo_base64 = load_logo(base_path / "images/logo_header.png")
    
    # Sidebar Logo and Navigation
    st.sidebar.markdown(
        f"""
        <div style="text-align: center;">
            <img src="data:image/png;base64,{logo_base64}" alt="Logo" style="width: 120px; margin-bottom: 20px;">
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Initialize session state for page tracking
    if 'page' not in st.session_state:
        st.session_state.page = 'Home'
    
    # Custom CSS to style buttons in the sidebar
    st.sidebar.markdown("""
        <style>
        /* Button styling */
        section[data-testid="stSidebar"] div.stButton > button {
        background-color: #3a60b2 !important;
        color: white !important;
        width: 100%; /* Make button take full width of sidebar */
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;      
        }
        section[data-testid="stSidebar"] div.stButton > button:hover {
            background-color: #0056b3 !important;
        }
        /* Change sidebar background to white */
        section[data-testid="stSidebar"] {
            background-color: white !important;
        }
        [data-testid="stSidebarCollapseButton"] button {
            color: #3a60b2 !important;
            }
        [data-testid="stSidebarCollapsedControl"] button {
            color: #3a60b2 !important;
            }

        </style>
    """, unsafe_allow_html=True)
    
    # Sidebar buttons for navigation
    st.sidebar.title("Paper Review Aggregator")
    
    if st.sidebar.button("Home"):
        st.session_state.page = "Home" 
    if st.sidebar.button("Review Aggregation"):
        st.session_state.page = "Review Aggregation"
    if st.sidebar.button("Services"):
        st.session_state.page = "Meta Reviewer Dashboard"
    if st.sidebar.button("Contact"):
        st.session_state.page = "Contact"
    if st.sidebar.button("About"):
        st.session_state.page = "About"
    
    # Display content based on the current page
    if st.session_state.page == "Home":
        home_page.home_page(custom_css)         
    elif st.session_state.page == "Review Aggregation":
        landing_page.landing_page(custom_css)
    elif st.session_state.page == "Meta Reviewer Dashboard":
        main_page.main_page(custom_css)
    elif st.session_state.page == "Contact":
        contact.show_contact_info(custom_css)
    elif st.session_state.page == "About":
        about.about_page(custom_css)
        

show_navigation_bar_and_content()


