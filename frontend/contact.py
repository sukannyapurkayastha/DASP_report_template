import modules.contact_info
from modules.shared_methods import use_default_container
import streamlit as st


def show_contact_info(custom_css):
    
    # Apply custom CSS Styles
    # st.markdown(custom_css, unsafe_allow_html=True)
    
    #show contact details
    use_default_container(modules.contact_info.show_contact_info)