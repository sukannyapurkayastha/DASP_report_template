# about.py

"""
About Page Module

This module defines the `about_page` function, which renders the About page of the
Paper Review Aggregator application. It applies custom CSS styles and displays
about content along with contact information using shared methods.
"""

import modules.contact_info
import modules.about_content
from modules.shared_methods import use_default_container
import streamlit as st


def about_page(custom_css):
    """
    Render the About page with custom styling and content.
    
    This function applies custom CSS styles to the Streamlit application and displays
    the about content and contact information by invoking corresponding functions from
    the `about_content` and `contact_info` modules.
    
    Parameters:
        custom_css (str): A string containing CSS styles to customize the appearance
                          of the Streamlit application.
    """
    
    # Apply custom CSS Styles
    st.markdown(custom_css, unsafe_allow_html=True)
    
    # Show about details
    use_default_container(modules.about_content.show_about_content)
    
    # Show contact details
    use_default_container(modules.contact_info.show_contact_info)
