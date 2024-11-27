import streamlit as st
from streamlit_extras.stylable_container import stylable_container
import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pickle
import modules.overview
import modules.attitude_roots
import modules.request_information
import modules.summary
import modules.contact_info
import modules.slideshow as ss
from modules.shared_methods import use_default_container 

#%% global variables


# Import data

try:
    overview = pd.read_csv(os.path.join("frontend/dummy_data", "dummy_overview.csv"), sep=";", encoding="utf-8")
    attitude_roots = pd.read_csv(os.path.join("frontend/dummy_data", "dummy_attitude_roots.csv"), sep=";", encoding="utf-8")
    #request_information = pd.read_csv(os.path.join("frontend/dummy_data", "dummy_request_information_list.csv"), sep=";", encoding="utf-8")
    with open(os.path.join('frontend', 'dummy_data', 'dummy_request_information.pkl'), 'rb') as file:
        request_information = pickle.load(file)

    summary = pd.read_csv(os.path.join("frontend/dummy_data", "dummy_summary.csv"), sep=";", encoding="utf-8")
except:
    FileNotFoundError("Some files have not been found.")
 

   
#%%% Set the page configuration

def main_page(custom_css):
    
    # Apply custom CSS Styles
    st.markdown(custom_css, unsafe_allow_html=True)
    
    st.title("Paper Review Summary")
    
    use_default_container(modules.overview.show_overview, overview)
    
    
    attitude_root_container = lambda: modules.attitude_roots.show_attitude_roots_data(attitude_roots)
    request_information_container = lambda: modules.request_information.show_request_information_data(request_information)
    summary_container = lambda: modules.summary.show_summary_data(summary)
    
    slideshow = ss.StreamlitSlideshow([attitude_root_container, request_information_container, summary_container], ["Attitude Roots", "Request Information", "Summary"])
    use_default_container(slideshow.show)
        
        
            
    use_default_container(modules.contact_info.show_contact_info)