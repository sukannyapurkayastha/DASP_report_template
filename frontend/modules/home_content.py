import streamlit as st

def show_home_content():
    st.title("Welcome to Paper Review Aggregator")
    st.write("Great that you are here. Unfortunatly your the only one around here.")
    st.write("You may explore our File Upload or some Preview,")

    
def show_home_teaser():
    
    # Main container with default Streamlit styling
    st.title("Discover the Power Today!")
    st.subheader("Get an aggragtion of your reviews to speed up meta review drastically!")
    
    # Add custom CSS for scrollable container
    st.markdown("""
        <style>
            .step-box {
                background-color: #bcd7f2;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 20px;
                box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
            }
        </style>
    """, unsafe_allow_html=True)
    
    
    # Render each step inside the scrollable container
    steps = [
        ("01", "Upload File or Enter Link", "Choose a file from your device or enter a openreview link to the thread you want to analyze."),
        ("02", "Start Analysis", "Click the submit button to begin the aggregation process for your the paper."),
        ("03", "Review Summary", "Receive a detailed summary of the paper's reviews and key points after the analysis is complete.")
    ]
    
    for number, title, description in steps:
        st.markdown(
            f"""
            <div class="step-box">
                <h2>{number}. {title}</h2>
                <p>{description}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
    def change_to_upload():
        st.session_state.page = "File Upload"
    def change_to_preview():
        st.session_state.page = "Meta Reviewer Dashboard"
    
        
    
    st.button("Go to File Upload", on_click= change_to_upload)
    st.button("Go to Preview", on_click= change_to_preview)
