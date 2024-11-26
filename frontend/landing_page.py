import streamlit as st

#%% Backend Logic

def switch_to_main_page():
    url_or_file = get_input()
    if valid_url_or_file(url_or_file):
        apply_backend_logic(url_or_file)
        st.session_state["current_page"] = "main_page"
    else:
        st.error("Invalid input! Please upload a file or provide a valid URL.")

# Dummy backend logic for now
def apply_backend_logic(url_or_file):
    st.session_state["backend_result"] = f"Processing: {url_or_file}"

# Validate if URL or file input is correct
def valid_url_or_file(url_or_file):
    return bool(url_or_file)

# Retrieve user input (uploaded file or entered URL)
def get_input():
    uploaded_file = st.session_state.get("uploaded_file")
    entered_url = st.session_state.get("entered_url")
    return uploaded_file if uploaded_file else entered_url

#%% CSS Styling
custom_css = """
<style>
    body {
        background-color: #f8f9fa;
    }
    .centered-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        margin-top: 30px;
    }
    .upload-box {
        border: 2px dashed #007bff;
        padding: 20px;
        border-radius: 10px;
        width: 80%;
        max-width: 500px;
        text-align: center;
        margin-bottom: 20px;
        background-color: #e9f4ff;
    }
    .btn-primary {
        background-color: #007bff;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
        cursor: pointer;
    }
    .btn-primary:hover {
        background-color: #0056b3;
    }
    .description-text {
        color: #555;
        margin-bottom: 20px;
        font-size: 16px;
    }
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
    # Set page configuration
    st.set_page_config(
        page_title="Paper Review Generator",
        page_icon="📄",
        layout="centered"
    )
    
    # Apply custom CSS
    st.markdown(custom_css, unsafe_allow_html=True)
    
    # Main container for alignment
    with st.container():
        st.title("Paper Review Summary")
        
        # Description text
        st.markdown(
            '<p class="description-text">'
            "Lorem ipsum dolor sit amet, <strong>consectetur adipiscing elit</strong>, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
            "At vero eos et accusam et justo duo dolores et ea rebum."
            "</p>",
            unsafe_allow_html=True,
        )
        
        # File uploader and URL input
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Select a file or drop it here", type=["txt", "pdf"])
        if uploaded_file:
            st.session_state["uploaded_file"] = uploaded_file
            st.success(f"Uploaded: {uploaded_file.name}")
        st.markdown('</div>', unsafe_allow_html=True)

        entered_url = st.text_input("Enter URL to Paper Reviews")
        if entered_url:
            st.session_state["entered_url"] = entered_url
            st.success(f"URL Entered: {entered_url}")
        
        # Submit button
        if st.button("Show Analysis"):
            switch_to_main_page()

# Run the landing page
if __name__ == "__main__":
    landing_page(custom_css)
