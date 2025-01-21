# contact_info.py

"""
Contact Information Module

This module defines functions to display contact information sections within the
Paper Review Aggregator application. It includes functionalities to load and
encode logo images, and to render contact details such as email and office
information using Streamlit.
"""

import streamlit as st
import base64
from pathlib import Path


def load_logo(filepath):
    """
    Load and encode an image file to a Base64 string for embedding in HTML.

    This function reads an image from the specified filepath, encodes it using Base64,
    and returns the encoded string, which can be used to embed the image in HTML content.

    Parameters:
        filepath (Path): The path to the image file to be loaded and encoded.

    Returns:
        str: The Base64-encoded string of the image.
    """
    with open(filepath, "rb") as logo_file:
        return base64.b64encode(logo_file.read()).decode("utf-8")


def show_contact_info():
    """
    Display the contact information section with embedded images.

    This function renders a section titled "Contact Us" with contact details for email
    and office locations. It uses Base64-encoded images for visual representation and
    organizes the information into two columns for better layout and readability.
    """
    # Base path for images
    base_path = Path(__file__).parent.parent

    # Load images as base64 strings
    email_logo_base64 = load_logo(base_path / "images/email.png")
    office_logo_base64 = load_logo(base_path / "images/office.png")

    # Section Title
    st.markdown("## Any issues? Get in touch with us!")
    st.markdown("### Contact Us")
    st.markdown("Lorem ipsum dolor sit amet, consectetur adipiscing elit.")

    # Create columns for contact information
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"""
            <div style="text-align: center;">
                <img src="data:image/png;base64,{email_logo_base64}" alt="Email" style="width:100px;" />
                <h3>Email</h3>
                <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse varius enim in eros.</p>
                <strong>info@paperreviewsummary.com</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div style="text-align: center;">
                <img src="data:image/png;base64,{office_logo_base64}" alt="Office" style="width:100px;" />
                <h3>Office</h3>
                <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse varius enim in eros.</p>
                <strong>123 Main Street, City, Country</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="invisbible-line-big">  </div>', unsafe_allow_html=True)
