# contact.py

"""
Contact Module

This module defines the function to display contact information within the
Paper Review Aggregator application. It utilizes shared methods and imports
functionalities from other modules to render contact details on the interface.
"""

import modules.contact_info
from modules.shared_methods import use_default_container
import streamlit as st


def show_contact_info(custom_css):
    """
    Render the contact information section with custom styling.
    
    This function applies custom CSS styles to the Streamlit application and
    displays contact details by invoking the `show_contact_info` function from
    the `modules.contact_info` module.
    
    Parameters:
        custom_css (str): A string containing CSS styles to customize the appearance
                          of the Streamlit application.
    """
    
    # Apply custom CSS Styles
    st.markdown(custom_css, unsafe_allow_html=True)
    
    # Show contact details
    use_default_container(modules.contact_info.show_contact_info)
