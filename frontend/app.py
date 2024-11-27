import streamlit as st
from landing_page import landing_page
from main_page import main_page

st.set_page_config(
    page_title="Paper Review Generator",
    page_icon=os.path.join("frontend/images" ,"logo.png")
    )

custom_css = """
    <style>
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
        section[data-testid="stSidebar"] div.stButton > button {
        background-color: #3a60b2 !important;
        color: white !important;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
        cursor: pointer;       
        }
        section[data-testid="stSidebar"] div.stButton > button:hover {
            background-color: #0056b3 !important;
        }
        /* Button styling */

        </style>
    """, unsafe_allow_html=True)
    
    # Sidebar buttons for navigation
    st.sidebar.title("Welcome")
    
    if st.sidebar.button("Home"):
        st.session_state.page = "Home"
    if st.sidebar.button("Services"):
        st.session_state.page = "About"
    if st.sidebar.button("Contact"):
        st.session_state.page = "Contact"
    
    # Display content based on the current page
    if st.session_state.page == "Home":
        landing_page.landing_page(custom_css)
    elif st.session_state.page == "About":
        main_page.main_page(custom_css)
    elif st.session_state.page == "Contact":
        contact.show_contact_info(custom_css)

show_navigation_bar_and_content()

