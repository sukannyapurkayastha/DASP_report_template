# app.py

"""
Paper Review Generator Streamlit Application

This application provides a user interface for aggregating and generating summaries
of paper reviews. It includes navigation between different pages such as Home,
Review Aggregation, Contact, and About. The application is styled using custom CSS
and incorporates logos and other UI elements to enhance user experience.
"""

import streamlit as st
import base64
from pathlib import Path
import main_page
import landing_page
import home_page
import os
import contact
import about

# Configure the Streamlit page
st.set_page_config(
    page_title="Paper Review Generator",
    page_icon=os.path.join("images", "logo.png")
)

# Custom CSS for styling the application
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

def safe_delete_session_state(key):
    """
    Deletes a key from the Streamlit session state.

    This function ensures that specific variables are removed from the session state
    when a user navigates away from a page, preventing unintended data persistence.

    Parameters:
        key (str): The key in the session state to be deleted.
    """
    if key in st.session_state:
        del st.session_state[key]

def show_navigation_bar_and_content():
    """
    Displays the navigation sidebar and renders content based on the selected page.

    This function handles the loading of the logo, setting up the sidebar with navigation
    buttons, and rendering the appropriate page content based on the user's selection.
    It also applies custom CSS for styling various UI elements.

    The supported pages include:
        - Home
        - Review Aggregation
        - Contact
        - About

    The function ensures that session state variables are appropriately managed when
    navigating between pages.
    """

    def load_logo(filepath):
        """
        Loads and encodes a logo image to Base64.

        This nested function reads an image file from the specified filepath and encodes
        it in Base64 format, which is suitable for embedding in HTML.

        Parameters:
            filepath (Path): The path to the logo image file.

        Returns:
            str: The Base64-encoded string of the logo image.
        """
        with open(filepath, "rb") as logo_file:
            return base64.b64encode(logo_file.read()).decode("utf-8")

    # Path to the local logo file
    base_path = Path(__file__).parent
    logo_base64 = load_logo(base_path / "images/logo_header.png")

    # Display the logo in the sidebar
    st.sidebar.markdown(
        f"""
        <div style="text-align: center;">
            <img src="data:image/png;base64,{logo_base64}" alt="Logo" style="width: 120px; margin-bottom: 20px;">
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Initialize session state for page tracking if not already set
    if 'page' not in st.session_state:
        st.session_state.page = 'Home'

    # Apply custom CSS to style buttons in the sidebar
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

    # Sidebar title and navigation buttons
    st.sidebar.title("Paper Review Aggregator")

    if st.sidebar.button("Home"):
        st.session_state.page = "Home"
    if st.sidebar.button("Review Aggregation"):
        st.session_state.page = "Review Aggregation"
    #if st.sidebar.button("Meta Reviewer Dashboard"):
    #    st.session_state.page = "Meta Reviewer Dashboard"
    if st.sidebar.button("Contact"):
        st.session_state.page = "Contact"
    if st.sidebar.button("About"):
        st.session_state.page = "About"

    # Render content based on the selected page
    if st.session_state.page == "Home":
        safe_delete_session_state("main_page_variables")
        st.session_state.page = home_page.home_page(custom_css)
    elif st.session_state.page == "Review Aggregation":
        safe_delete_session_state("main_page_variables")
        landing_page.landing_page(custom_css)
    elif st.session_state.page == "Meta Reviewer Dashboard":
        main_page.main_page(custom_css)
    elif st.session_state.page == "Contact":
        safe_delete_session_state("main_page_variables")
        contact.show_contact_info(custom_css)
    elif st.session_state.page == "About":
        safe_delete_session_state("main_page_variables")
        about.about_page(custom_css)

# Execute the navigation and content display
show_navigation_bar_and_content()
