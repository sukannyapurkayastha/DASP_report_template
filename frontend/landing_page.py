import streamlit as st
from streamlit_extras.stylable_container import stylable_container
import os

#%% global variables

invalid_url_or_file = False


#%% Button + backendlogic

def switch_to_main_page():
    url_or_file = get_input()
    if valid_url_or_file(url_or_file):
        apply_backend_logic(url_or_file)
        st.session_state["current_page"] = "main_page"
    else:
        st.write("Invalid input!")

#TODO: Finally we need to apply backend scripts here so we can show the results on main_page
def apply_backend_logic(url_or_file):
    return None
#TODO: Check wheter the given url or file is correct
def valid_url_or_file(url_or_file):
    return True

# TODO: Get the URL or File the user has uploaded
def get_input():
    return ""


#%% Landing-Page itself


def landing_page(custom_css):
    
    #%%% Set the page configuration
    st.set_page_config(
        page_title="Paper Review Generator",
        page_icon=os.path.join("logo.png")
        )
    
    # Apply custom CSS Styles
    
    st.markdown(custom_css, unsafe_allow_html=True)
    
        
    st.title("Paper Review Generator")
    
    # Callback to switch pages
        
    # Create Submit-Button
    st.button("Submit", on_click = switch_to_main_page)

        