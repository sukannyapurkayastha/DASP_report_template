import streamlit as st
from modules.shared_methods import use_default_container 
import modules.contact_info
from pathlib import Path

#%% Backend Logic

def switch_to_main_page(skip_validation=False):
    if skip_validation:
        # Directly switch to the main page without validation
        url_or_file = get_input()
        apply_backend_logic(url_or_file)
        st.session_state["current_page"] = "main_page"
        st.session_state["backend_result"] = "Skipped validation and switched to main page."
    else:
        url_or_file = get_input()
        if valid_url_or_file(url_or_file):
            apply_backend_logic(url_or_file)
            st.session_state["current_page"] = "main_page"
        else:
            st.error("Invalid input! Please upload a file or provide a valid URL.")


def apply_backend_logic(url_or_file):
    st.session_state["backend_result"] = f"Processing: {url_or_file}"


def valid_url_or_file(url_or_file):
    return bool(url_or_file)

def get_input():
    uploaded_file = st.session_state.get("uploaded_file")
    entered_url = st.session_state.get("entered_url")
    return uploaded_file if uploaded_file else entered_url

    


#%% Landing Page Function
def landing_page(custom_css):
    
    def provide_sample_download():
        
        # Base path for images
        base_path = Path(__file__).parent

        # Path to your .docx file
        file_path = Path(base_path / "dummy_data/sample_reviews.docx")
        
        # Load the .docx file into memory
        with open(file_path, "rb") as file:
            docx_file = file.read()
        
        # Provide a download button
        col1, col2 = st.columns([2.4,1])
        with col2:
            st.download_button(
                label="Download Sample DOCX File",
                data=docx_file,
                file_name="sample_reviews.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
    
    def content():
        # Apply custom CSS
        st.markdown(custom_css, unsafe_allow_html=True)
    
        st.title("Paper Review Aggregator")
        
        st.write("You can either provide a link of a openreview thread for the desired paper review aggregation (account of openreview login credentials required) or you provide us a file containing all reviews to aggregate. In this case you must use our template format.")
        
        # Create tabs
        tab1, tab2 = st.tabs(["Enter URL", "Upload file"])
        
        # Web API
        with tab1:
            st.write("To provide the aggregation of your desired paper we need your openreview login creditals and a valid link to the desired reviews that will be aggregated.")
            st.write("In case you don't have an openreview account you can either create one at openreview.com or upload a file instead (use tab above).")
            col1, col2 = st.columns([1,2])
            with col1:
                # Create the username field
                username = st.text_input("Openreview Username")
                
                # Create the password field
                password = st.text_input("Openreview Password", type="password")
            
            with col2:
                st.markdown('<div class="invisbible-line-small">  </div>', unsafe_allow_html=True)
                # Button to submit the credentials
                if st.button("Login"):
                    if username == "admin" and password == "1234":  # TODO: create function that validates login credentials
                        st.success("Login successful!")
                    else:
                        st.error("Invalid username or password.")
            st.text_input("Enter URL to Paper Reviews", key="entered_url", placeholder="https://openreview.net/forum?id=XXXXXXXXX")
        
        # File Uploader
        with tab2:
            st.write("In case you cannot provide a URL you can also uplaod a docx file containing all reviews. To do so please download a sample file and provide your data in this format.")
            provide_sample_download()
            uploaded_file = st.file_uploader("Select a file or drop it here to upload it.", type=["docx"])
            if uploaded_file:
                st.session_state["uploaded_file"] = uploaded_file
                st.success(f"Uploaded: {uploaded_file.name}")
        
        
        col1, col2 = st.columns([4.5,1])
        with col2:
            if st.button("Show Analysis"):
                switch_to_main_page()
    
        #st.button("Go to analysis", on_click=lambda: switch_to_main_page(skip_validation=True))

    use_default_container(content)
    
    use_default_container(modules.contact_info.show_contact_info)