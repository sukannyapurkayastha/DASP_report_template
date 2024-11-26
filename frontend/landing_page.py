import streamlit as st

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

#%% CSS Styling
landing_page_css = """
<style>
/* General page background */
body {
    background-color: #f0f0f0; /* Light grey background */
}

/* Main content area (Streamlit's block container) */
.block-container {
    border-radius: 0.5rem;
    padding: calc(1em - 1px);
    width: 95%; 
    max-width: 800px; 
    margin: 20px auto; 
    background-color: white; 
    display: flex;
    flex-direction: column;
    gap: 20px; 

                
}

/* Description text styling */
.description-text {
    color: #555;
    font-size: 16px;
}

/* File uploader and URL input */
.stFileUploader {
    margin-bottom: 20px;
}
.stTextInput input {
    color: #333 !important;
    font-size: 16px;
    border: 2px solid #007bff !important; /* Blue border */
    border-radius: 5px;
    width: 100%; /* Ensure input takes up full width */
    padding: 10px;
}
.stTextInput {
    margin-bottom: 20px;
}

/* Button styling */
.stButton button {
    background-color: #007bff !important;
    color: white !important;
    border: none;
    border-radius: 5px;
    padding: 10px 20px;
    font-size: 16px;
    cursor: pointer;
}
.stButton button:hover {
    background-color: #0056b3 !important;
}
</style>
"""
#%% Landing Page Function
def landing_page(custom_css):
    st.set_page_config(
        page_title="Paper Review Generator",
        page_icon="📄",
        layout="centered"
    )
    
    # Apply custom CSS
    st.markdown(landing_page_css, unsafe_allow_html=True)

    st.title("Paper Review Summary")
    
    st.markdown(
        '<p class="description-text">'
        "Lorem ipsum dolor sit amet, <strong>consectetur adipiscing elit</strong>, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
        "At vero eos et accusam et justo duo dolores et ea rebum."
        "</p>",
        unsafe_allow_html=True,
    )
    
    # File uploader
    uploaded_file = st.file_uploader("Select a file or drop it here", type=["txt", "pdf"])
    if uploaded_file:
        st.session_state["uploaded_file"] = uploaded_file
        st.success(f"Uploaded: {uploaded_file.name}")
    

    st.text_input("Enter URL to Paper Reviews", key="entered_url")
    

    if st.button("Show Analysis"):
        switch_to_main_page()

    st.button("Go to analysis", on_click=lambda: switch_to_main_page(skip_validation=True))


if __name__ == "__main__":
    landing_page(landing_page_css)
