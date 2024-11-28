import modules.contact_info
import modules.home_content
from modules.shared_methods import use_default_container
import streamlit as st


def home_page(custom_css):
    
    # Apply custom CSS Styles
    st.markdown(custom_css, unsafe_allow_html=True)
    
    #show home details
    use_default_container(modules.home_content.show_home_content)
    
    #show contact details
    use_default_container(modules.contact_info.show_contact_info)