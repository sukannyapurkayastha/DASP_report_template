import json

import openreview
from requests.exceptions import ConnectionError
import streamlit as st

from modules.shared_methods import use_default_container
import modules.contact_info
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# Import from backend because we are importing from another module from root and have to go up a directory level
import sys
import os
import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frontend.clients import OpenReviewClient, UploadedFileProcessor


# %% Backend Logic


def switch_to_main_page(skip_validation=False):
    if skip_validation:
        # Directly switch to the main page without validation
        url_or_file = get_input()
        # apply_backend_logic(url_or_file)
        st.session_state.page = "Meta Reviewer Dashboard"
        st.session_state["backend_result"] = "Skipped validation and switched to main page."
    else:
        url_or_file = get_input()
        if valid_url_or_file(url_or_file):
            st.session_state.page = "Meta Reviewer Dashboard"
            st.rerun()
        else:
            st.error("Invalid input! Please upload a file or provide a valid URL.")


def valid_url_or_file(url_or_file):
    return bool(url_or_file)


def get_input():
    reviews = st.session_state.get("reviews")
    return reviews


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
        file_path = Path(base_path / "data/review_template.docx")

        # Load the .docx file into memory
        with open(file_path, "rb") as file:
            docx_file = file.read()

        # Provide a download button
        col1, col2 = st.columns([2.4, 1])
        with col2:
            st.download_button(
                label="Download Sample DOCX File",
                data=docx_file,
                file_name="review.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )

    def content():
        # Apply custom CSS
        # st.markdown(custom_css, unsafe_allow_html=True)

        st.title("Paper Review Aggregator")

        st.write(
            "You can either provide a link of a openreview thread for the desired paper review aggregation (account of openreview login credentials required) or you provide us a file containing all reviews to aggregate. In this case you must use our template format.")

        # Create tabs
        # tab1, tab2 = st.tabs(["Enter URL", "Upload file"])

        tabs = ["Enter URL", "Upload file"]
        tab1, tab2 = st.tabs(tabs)

        # Function to display the "Show Analysis" button
        def display_show_analysis_button(key):
            col1, col2 = st.columns([4.5, 1])
            with col2:
                if st.button("Show Analysis", key=key):
                    print("pressed show analysis button")
                    switch_to_main_page()

        # Function to update the active tab in session state
        def set_active_tab(tab_name):
            st.session_state['active_tab'] = tab_name

        # Initialize session state for active tab
        if 'active_tab' not in st.session_state:
            st.session_state['active_tab'] = tabs[0]

        # Web API
        with tab1:
            set_active_tab("Enter URL")

            st.write(
                "To provide the aggregation of your desired paper we need your openreview login creditals and a valid link to the desired reviews that will be aggregated.")
            st.write(
                "In case you don't have an openreview account you can either create one at openreview.com or upload a file instead (use tab above).")

            # Session states
            if 'logged_in' not in st.session_state:
                st.session_state['logged_in'] = False
            if 'client' not in st.session_state:
                st.session_state['client'] = None

            if not st.session_state['logged_in']:
                col1, col2 = st.columns([1, 2])
                with col1:
                    # Create the username field
                    username = st.text_input("Username (Openreview)")

                    # Create the password field
                    password = st.text_input("Password (Openreview)", type="password")

                with col2:
                    st.markdown('<div class="invisbible-line-small">  </div>', unsafe_allow_html=True)
                    # Button to submit the credentials
                    if st.button("Login"):
                        try:
                            client = OpenReviewClient(username=username, password=password)
                            st.success("Login successful!")
                            st.session_state["logged_in"] = True
                            st.session_state["client"] = client
                            # Rerun the app to update the UI
                            st.rerun()
                        except openreview.OpenReviewException as ore:
                            error_response = ore.args[0] if ore.args else "An error occurred with OpenReview."
                            if "Invalid username or password" in error_response.get('message', ''):
                                st.error("Invalid username or password. Please try again.")
                            else:
                                st.error(
                                    f"An unexpected error occurred: {error_response.get('message', 'No details available')}")
                        except ConnectionError:
                            st.error(
                                "Connection error: Unable to connect to OpenReview. Please check your internet connection.")
                        except requests.exceptions.RequestException as e:
                            if "Max retries exceeded" in str(e) or "Failed to establish a new connection" in str(e):
                                st.error(
                                    "Connection error: Unable to connect to OpenReview. Please check your internet connection.")
                            else:
                                st.error(f"An unexpected error occurred: {str(e)}")
                        except Exception as e:
                            st.error("An unexpected error occurred. Please try again later.")
                            st.error(e)


            else:
                client = st.session_state['client']
                st.success(f"Welcome {client.client.user['user']['profile']['fullname']}!")
                input_url = st.text_input("Enter URL to Paper Reviews", key="entered_url",
                                          placeholder="https://openreview.net/forum?id=XXXXXXXXX")

                if input_url:
                    paper_id = extract_paper_id(input_url)
                    if paper_id:
                        st.session_state["paper_id"] = paper_id

                        try:
                            paper = client.get_reviews_from_id(paper_id)
                            # TODO: if we pass url or file later in show_analysis, then we don't need to get reviews right now.
                            st.session_state["reviews"] = paper.reviews
                            st.success(f'Reviews extracted from paper: "{paper.title}"')
                        except Exception as e:
                            st.error(e)
                    else:
                        st.error("Invalid OpenReview URL.")
                else:
                    st.info("Please enter a valid OpenReview URL to proceed.")

            # Check conditions to display the "Show Analysis" button
            if ('reviews' in st.session_state and st.session_state['reviews'] and st.session_state[
                'active_tab'] == "Enter URL"):
                display_show_analysis_button(key="show_analysis_button_tab1")

        # File Uploader
        with tab2:
            set_active_tab("Upload file")

            st.write(
                "In case you cannot provide a URL you can also upload a docx file containing all reviews. "
                "To do so, please download a sample file and provide your data in this format."
            )
            provide_sample_download()  # template download

            try:
                uploaded_files = st.file_uploader(
                    "Select a file or drop it here to upload it.",
                    type=["docx"],
                    accept_multiple_files=True
                )

                if uploaded_files:
                    st.session_state["uploaded_files"] = uploaded_files
                    num_files = len(uploaded_files)
                    file_word = "file" if num_files == 1 else "files"
                    st.success(f"Uploaded {num_files} {file_word} successfully.")

                    for uploaded_file in uploaded_files:
                        try:
                            # Process each uploaded file
                            upload_processor = UploadedFileProcessor([uploaded_file])
                            reviews = upload_processor.process()
                            st.session_state["reviews"] = reviews

                            # Additional success feedback
                            st.info(f"Processed {uploaded_file.name} successfully.")
                        except ValueError as ve:
                            st.error(f"File {uploaded_file.name} has invalid content: {ve}")
                        except FileNotFoundError:
                            st.error(f"File {uploaded_file.name} could not be found.")
                        except Exception as e:
                            st.error(f"An unexpected error occurred while processing {uploaded_file.name}.")


                else:
                    st.info("Please upload at least one DOCX file to proceed.")

            except Exception as e:
                st.error("An error occurred while uploading files. Please try again.")

            # Check conditions to display the "Show Analysis" button
            if ('reviews' in st.session_state and st.session_state['reviews'] and
                    st.session_state['active_tab'] == "Upload file"):
                display_show_analysis_button("show_analysis_button_tab2")

        # Optionally, provide an informational message when no reviews are available
        # if not ('reviews' in st.session_state and st.session_state['reviews']):
        #     st.info("Please provide a valid URL or upload a file to proceed to the analysis.")

        # st.button("Go to analysis", on_click=lambda: switch_to_main_page(skip_validation=True))

    use_default_container(content)

    use_default_container(modules.contact_info.show_contact_info)
