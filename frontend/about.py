import modules.contact_info
import modules.about_content
from modules.shared_methods import use_default_container
import streamlit as st


def about_page(custom_css):
    
    # Apply custom CSS Styles
    st.markdown(custom_css, unsafe_allow_html=True)
    
    #show about details
    use_default_container(modules.about_content.show_about_content)
    
    
    #show contact details
    use_default_container(modules.contact_info.show_contact_info)