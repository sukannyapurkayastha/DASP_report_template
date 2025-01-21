# attitude_roots.py

"""
Attitude Roots Module

This module defines the `show_attitude_roots_data` function, which displays
attitude roots information within the Paper Review Aggregator application. It
renders progress bars and comments associated with each attitude root, providing
a visual representation of the data.
"""

import streamlit as st
from modules.shared_methods import select_color_attitude_or_request, draw_progress_bar, show_comments


def show_attitude_roots_data(attitude_roots_data):
    """
    Display attitude roots data with progress bars and expandable comments.
    
    This function renders each attitude root with a corresponding progress bar indicating
    its proportion. Additionally, it provides an expandable section for comments related
    to each attitude root.
    
    Args:
        attitude_roots_data (pd.DataFrame): A DataFrame containing attitude roots information.
    """
    
    #%% Helper methods
    
    def show_header_with_progress(row, desc):
        """
        Display the header with a progress bar for a given row.
        
        This helper function calculates the percentage based on the fraction, selects an
        appropriate color, and renders a progress bar. If a description is provided,
        it also displays the description.
        
        Args:
            row (pd.Series): A row from the DataFrame containing attitude root information.
            desc (bool): Flag indicating whether to display the description.
        """
        try: 
            fraction = eval(row[1])  # TODO: the fraction should be a number --> remove eval
        except:
            fraction = row[1]
        percentage = int(fraction * 100)  # Convert fraction to percentage
        color = select_color_attitude_or_request(fraction)
        draw_progress_bar(color, percentage)  # draws the progressbar
        if desc: 
            st.markdown(f"<h4 style='font-size:12px; margin: 0px; padding: 0px; text-align: right;'>{row[2]}</h4>", unsafe_allow_html=True)
        else:
            st.markdown('<div class="invisbible-line-minor">  </div>', unsafe_allow_html=True)
            
    #%% Actual attitude root generation
    with st.container():
        st.markdown('<div class="invisbible-line-minor">  </div>', unsafe_allow_html=True)
        if attitude_roots_data.empty:
            st.markdown("<h4 style='font-size:16px; margin: 0px; padding: 0px;'>No Attitude Roots were found.</h4>", unsafe_allow_html=True)
        else:
            for index, row in attitude_roots_data.iterrows():
                col1, col2, col3 = st.columns([2, 7, 0.2])
                with col1:
                    st.markdown(f"<h4 style='font-size:16px; margin: 0px; padding: 0px;'>{row[0]}</h4>", unsafe_allow_html=True)
                with col2:
                    show_header_with_progress(row, True)
                with st.expander("Comments"):
                    show_comments(row)
                st.markdown('<div class="invisbible-line-minor">  </div>', unsafe_allow_html=True)
