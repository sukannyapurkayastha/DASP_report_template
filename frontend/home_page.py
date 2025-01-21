# home_page.py

"""
Home Page Module

This module defines the `home_page` function, which renders the Home page of the
Paper Review Aggregator application. It applies custom CSS styles and displays
home content, teasers, and contact information using shared methods.
"""

import modules.contact_info
import modules.home_content
from modules.shared_methods import use_default_container
import streamlit as st


def home_page(custom_css):
    """
    Render the Home page with custom styling and content.

    This function applies custom CSS styles to the Streamlit application and displays
    the home content, teaser section, and contact information by invoking corresponding
    functions from the `home_content` and `contact_info` modules.

    Parameters:
        custom_css (str): A string containing CSS styles to customize the appearance
                          of the Streamlit application.
    """
    
    # Apply custom CSS Styles
    st.markdown(custom_css, unsafe_allow_html=True)
    
    # Show home details
    use_default_container(modules.home_content.show_home_content)
    
    # Show teaser
    use_default_container(modules.home_content.show_home_teaser)
    
    # Show contact details
    use_default_container(modules.contact_info.show_contact_info)
