import streamlit as st

def show_home_content():
    st.title("Welcome to Paper Review Aggregator")

    st.markdown('<div class="invisbible-line-minor">  </div>', unsafe_allow_html=True)    
    st.subheader("How meta reviews used to be")
    st.write("An essential part of the work of meta reviewers is to screen reviews from several reviewers and make a decision on the acceptance of the paper for the journal. This process can be very demanding. Well, it used to be, but not anymore. ")
    
    st.markdown('<div class="invisbible-line-minor">  </div>', unsafe_allow_html=True)
    st.subheader("What is this page about?")
    st.write('The aim of this project is to simplify the life of “meta-reviewers” drastically by summarizing all the reviews a paper has received so far. To do this, you simply need to follow the three simple steps below.')
    st.markdown('<div class="invisbible-line-small">  </div>', unsafe_allow_html=True)


def show_home_teaser():
    # Main container with default Streamlit styling
    st.title("Discover the Power of AI Today!")
    st.subheader("Get an aggregation of your reviews to speed up meta reviews drastically!")
    
    
    # Render each step with unique inline rotation
    steps = [
        ("01", "Upload a File or Enter a Link", "Choose a file from your device or enter an OpenReview link to the thread you want to analyze.", "rotate(0.5deg)"),
        ("02", "Start Aggregation", "Click the submit button to trigger an AI pipeline starting the aggregation process for your paper.", "rotate(-0.5deg)"),
        ("03", "Receive a powerful Summary", "Receive a detailed summary of the paper's reviews and key points after the analysis is complete.", "rotate(0.5deg)"),
    ]
    
    for number, title, description, rotation in steps:
        st.markdown(
            f"""
            <div class="step-box" style="transform: {rotation};">
                <h2>{number}. {title}</h2>
                <p>{description}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    # Navigation buttons
    def change_to_upload():
        st.session_state.page = "Review Aggregation"
    
    def change_to_preview():
        st.session_state.page = "Meta Reviewer Dashboard"
    
    st.markdown('<div class="invisbible-line-small">  </div>', unsafe_allow_html=True)
    col0, col1, col2, col3 = st.columns([2,2,2,2])
    with col1:
        st.button("Review Aggregation ", on_click=change_to_upload)
    with col2:
        st.button("Show Example", on_click=change_to_preview)
        
    st.markdown('<div class="invisbible-line-small">  </div>', unsafe_allow_html=True)
