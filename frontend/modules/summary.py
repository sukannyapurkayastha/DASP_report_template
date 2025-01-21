import streamlit as st
import pandas as pd


def show_summary_data(summary_data):
    with st.container():
        st.markdown('<div class="invisbible-line-minor">  </div>', unsafe_allow_html=True)
        if summary_data.empty:
            st.markdown("<h4 style='font-size:16px; margin: 0px; padding: 0px;'>No Summary was found.</h4>",
                        unsafe_allow_html=True)
        else:
            for index, row in summary_data.iterrows():
                st.markdown(row[0])  # write summary
                st.markdown("")
