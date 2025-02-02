# summary.py

"""
Summary Module

This module defines the `show_summary_data` function, which displays summary
information within the Paper Review Aggregator application. It presents the
summary data in a readable format, allowing users to review the aggregated
summaries of paper reviews.
"""

import streamlit as st


def show_summary_data(summary_data):
    """
    Display summary data within the Streamlit application.
    
    This function renders each summary entry from the provided DataFrame. If no
    summary data is available, it notifies the user accordingly.
    
    Args:
        summary_data (pd.DataFrame): A DataFrame containing summary information.
    """
    with st.container():
        st.markdown('<div class="invisbible-line-minor">  </div>', unsafe_allow_html=True)
        if summary_data.empty:
            st.markdown("<h4 style='font-size:16px; margin: 0px; padding: 0px;'>No Summary was found.</h4>",
                        unsafe_allow_html=True)
        else:
            for index, row in summary_data.iterrows():
                st.write(row[0])  # Write summary
            
            # add minor hint that AI makes failures
            st.markdown('<div class="invisbible-line-minor">  </div>', unsafe_allow_html=True)
            st.write("*Please note that these comments have been aggregated using AI. They may contain errors.*")    
            
