# contact_info.py
import streamlit as st
import base64
from pathlib import Path

def load_logo(filepath):
    """
    Loads an image file and encodes it as a base64 string for embedding.
    """
    with open(filepath, "rb") as logo_file:
        return base64.b64encode(logo_file.read()).decode("utf-8")

def show_contact_info():
    """
    Displays a contact information section with pictures for each item using base64-encoded images.
    """
    # Base path for images
    base_path = Path(__file__).parent.parent

    # Load images as base64 strings
    email_logo_base64 = load_logo(base_path / "images/email.png")
    office_logo_base64 = load_logo(base_path / "images/office.png")

    # Section Title
    st.markdown("## Get in touch with us today!")
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