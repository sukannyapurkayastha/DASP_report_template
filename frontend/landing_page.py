import streamlit as st
from modules.shared_methods import use_default_container
import modules.contact_info
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# Import from backend because we are importing from another module from root and have to go up a directory level
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.dataloading.loaders import OpenReviewLoader


# %% Backend Logic

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


def extract_paper_id(url):
    parsed_url = urlparse(url)
    if parsed_url.netloc != 'openreview.net':
        st.error("Invalid OpenReview URL.")
        return None
    if parsed_url.path != '/forum':
        st.error("Invalid OpenReview URL path.")
        return None
    query_params = parse_qs(parsed_url.query)
    paper_id_list = query_params.get('id', [])
    if paper_id_list:
        return paper_id_list[0]
    else:
        st.error("Paper ID not found in URL.")
        return None


# %% Landing Page Function
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
        col1, col2 = st.columns([2.4, 1])
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

        st.write(
            "You can either provide a link of a openreview thread for the desired paper review aggregation (account of openreview login credentials required) or you provide us a file containing all reviews to aggregate. In this case you must use our template format.")

        # Create tabs
        tab1, tab2 = st.tabs(["Enter URL", "Upload file"])

        # Web API
        with tab1:
            st.write(
                "To provide the aggregation of your desired paper we need your openreview login creditals and a valid link to the desired reviews that will be aggregated.")
            st.write(
                "In case you don't have an openreview account you can either create one at openreview.com or upload a file instead (use tab above).")

            # Session states
            if 'logged_in' not in st.session_state:
                st.session_state['logged_in'] = False
            if 'client' not in st.session_state:
                st.session_state['client'] = None

            if st.session_state['logged_in']:
                client = st.session_state['client']
                st.success(f"Welcome {client.client.user['user']['profile']['fullname']}!")
                input_url = st.text_input("Enter URL to Paper Reviews", key="entered_url",
                                          placeholder="https://openreview.net/forum?id=XXXXXXXXX")

                if input_url:
                    paper_id = extract_paper_id(input_url)
                    if paper_id:
                        st.session_state["paper_id"] = paper_id
                        # st.success(f"Paper ID extracted: {paper_id}")

                        try:
                            paper = client.get_paper_reviews(paper_id)
                            st.session_state["reviews"] = paper.reviews
                            st.success(f'Reviews extracted from paper: "{paper.title}"')
                        except Exception as e:
                            st.error(e)

            else:
                col1, col2 = st.columns([1, 2])
                with col1:
                    # Create the username field
                    username = st.text_input("Openreview Username")

                    # Create the password field
                    password = st.text_input("Openreview Password", type="password")

                with col2:
                    st.markdown('<div class="invisbible-line-small">  </div>', unsafe_allow_html=True)
                    # Button to submit the credentials
                    if st.button("Login"):
                        try:
                            client = OpenReviewLoader(username=username, password=password)
                            st.success("Login successful!")
                            st.session_state['logged_in'] = True
                            st.session_state['client'] = client
                            # Rerun the app to update the UI
                            st.rerun()
                        except Exception as e:
                            st.error(f"{e.args[0]['status']}: {str(e.args[0]['message'])}")

        # File Uploader
        with tab2:
            st.write(
                "In case you cannot provide a URL you can also uplaod a docx file containing all reviews. To do so please download a sample file and provide your data in this format.")
            provide_sample_download()
            uploaded_file = st.file_uploader("Select a file or drop it here to upload it.", type=["docx"])
            if uploaded_file:
                st.session_state["uploaded_file"] = uploaded_file
                st.success(f"Uploaded: {uploaded_file.name}")

        # Show the "Show Analysis" button only if a URL is provided (and reviews are extracted) or a file is uploaded
        if ('reviews' in st.session_state and st.session_state['reviews']) or 'uploaded_file' in st.session_state:
            col1, col2 = st.columns([4.5, 1])
            with col2:
                if st.button("Show Analysis"):
                    switch_to_main_page()
        else:
            st.info("Please provide a valid URL or upload a file to proceed to the analysis.")

        # st.button("Go to analysis", on_click=lambda: switch_to_main_page(skip_validation=True))

    use_default_container(content)

    use_default_container(modules.contact_info.show_contact_info)
