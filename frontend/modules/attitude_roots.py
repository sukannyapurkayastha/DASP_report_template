import streamlit as st
import pandas as pd
from modules.shared_methods import select_color_attitude_or_request, draw_progress_bar


def show_attitude_roots_data(attitude_roots_data):
    
    #%% Helper methods
    
    def show_reviewers_attitude_comments(row):
        i = 3
        while i < len(row) and isinstance(row[i], str):
            st.markdown(f'<div class="content-box">{row[i]}</div>', unsafe_allow_html=True)
            st.markdown('<div class="content-box"> </div>', unsafe_allow_html=True)
            i += 1
            
    def show_header_with_progress(row, desc):
        try: 
            fraction = eval(row[1])  # todo: the fraction should be a number --> remove eval
        except:
            fraction = row[1]
        percentage = int(fraction*100)  # Convert fraction to percentage
        color = select_color_attitude_or_request(fraction)
        draw_progress_bar(color, percentage) # draws the progressbar
        if desc: 
            st.markdown(f"<h4 style='font-size:12px; margin: 0px; padding: 0px; text-align: right;'>{row[2]}</h4>", unsafe_allow_html=True)
        else:
            st.markdown('<div class="invisbible-line-minor">  </div>', unsafe_allow_html=True)
        

        
    #%% actual attitude root generation
    with st.container():
        st.markdown('<div class="invisbible-line-minor">  </div>', unsafe_allow_html=True)
        if attitude_roots_data.empty:
            st.markdown("<h4 style='font-size:16px; margin: 0px; padding: 0px;'>No Attitude Roots were found.</h4>", unsafe_allow_html=True)
        else:
            for index, row in attitude_roots_data.iterrows():
                col1, col2 = st.columns([2.1,8])
                with col1:
                    st.markdown(f"<h4 style='font-size:16px; margin: 0px; padding: 0px;'>{row[0]}</h4>", unsafe_allow_html=True)
                with col2:
                    show_header_with_progress(row, True)
                with st.expander("Comments"):
                    show_reviewers_attitude_comments(row)
                st.markdown('<div class="invisbible-line-minor">  </div>', unsafe_allow_html=True)
                
        
    