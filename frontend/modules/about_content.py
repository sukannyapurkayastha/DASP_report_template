import streamlit as st

def show_about_content():
    st.title("About Us")
    st.write("Discover the inspiration and purpose behind **Paper Review Aggregator**, a project developed by students at **TU Darmstadt, Germany**.")
    st.markdown('<div class="invisbible-line-minor">  </div>', unsafe_allow_html=True)

    

    st.subheader("Our Mission")
    st.write(""" At **Paper Review Aggregator**, our mission is to simplify the work of meta-reviewers by
            providing an AI-driven platform that aggregates and summarizes paper reviews.
            By reducing manual effort and enhancing decision-making, we aim to streamline the peer-review process 
            for academic research.
                """)             
    st.markdown('<div class="invisbible-line-minor">  </div>', unsafe_allow_html=True)


    
    st.subheader("About the Project")
    st.write("""
            This application was developed as part of the **"Data Analysis Software Project"** course 
            at **Technische Universität Darmstadt, Germany**. The project allowed Master’s students to apply 
            their data analysis, software engineering, and machine learning skills to solve real-world problems.
             """)
    st.write(""" 
                 The platform demonstrates the effective use of AI in summarizing research reviews, helping 
                 researchers and decision-makers save time and effort during the academic peer-review process.
             """)
    st.markdown('<div class="invisbible-line-minor">  </div>', unsafe_allow_html=True)


    
    st.subheader("Our Vision")
    st.write(""" We envision a future where AI assists academic communities globally, making scholarly 
                publishing more efficient and accessible. By bridging the gap between reviewers and 
                decision-makers, we strive to create a transparent, unbiased, and seamless review process.
            """)
    st.markdown('<div class="invisbible-line-minor">  </div>', unsafe_allow_html=True)
  
    
  
    st.subheader("Importance of Academic Research")
    st.write(""" At Paper Review Aggregator, we recognize that academic research is only as impactful as its accessibility and utility. 
             By simplifying the peer-review process and enhancing the efficiency of decision-making, we aim to support the academic community
             in their mission to make the world a better place.
             """)
    st.write("""
             Whether it's a groundbreaking study or a meta-analysis of existing work, every piece of research deserves careful consideration
             and timely dissemination. Our platform is one small step toward a brighter, more informed future for humanity.
             """)
    st.markdown('<div class="invisbible-line-small">  </div>', unsafe_allow_html=True)

    

    st.subheader("The Team")
    st.write("""
                This project was developed by a team of Master’s students from TU Darmstadt, combining their expertise 
                in business informatics, cognitive science and computer science. Their shared goal was to create a 
                functional, user-friendly application that addresses a significant challenge in academia.
             """)
    st.write("""
                As part of the **Data Analysis Software Project** course, this project represents a culmination of 
                months of learning, coding, and collaboration. It may be further maintained or expanded in the future 
                by subsequent cohorts or contributors.
             """)
    st.markdown('<div class="invisbible-line-minor">  </div>', unsafe_allow_html=True)
