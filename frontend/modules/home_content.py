import streamlit as st

def show_home_content():
    st.title("Welcome to Paper Review Summary Aggregator")
    st.write("Great that you are here. Unfortunatly your the only one around here.")
    st.write("You may explore our File Upload or some Preview,")
    
    def change_to_upload():
        st.session_state.page = "File Upload"
    def change_to_preview():
        st.session_state.page = "Meta Reviewer Dashboard"
    
        
    
    st.button("Go to File Upload", on_click= change_to_upload)
    st.button("Go to Preview", on_click= change_to_preview)