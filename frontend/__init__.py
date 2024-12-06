# import streamlit as st
# from streamlit_extras.stylable_container import stylable_container
# import time
# import pandas as pd
# import os
#
#
# attitude_roots = pd.read_csv(os.path.join("dummy_data", "dummy_attitude_roots.csv"), sep=";", encoding="utf-8")
# request_information = pd.read_csv(os.path.join("dummy_data", "dummy_request_information.csv"), sep=";", encoding="utf-8")
# summary = pd.read_csv(os.path.join("dummy_data", "dummy_summary.csv"), sep=";", encoding="utf-8")
#
#
# # Page configuration
# st.set_page_config(page_title="Paper Review Summary")
#
# # CSS styles
# st.markdown("""
#     <style>
#     .header_carmen {
#         font-size: 108px;
#         font-weight: bold;
#     }
#     body {
#         background-color: #F2F2F2;
#     }
#     .content-box {
#         padding-left: 2px;
#         margin-bottom: 0px;
#         border-radius: 5px;
#         background-color: #E8E8E8;
#         margin-bottom: 2px;
#     }
#     .text-box {
#         padding-left: 2px;
#         margin-bottom: 0px;
#         border-radius: 5px;
#         background-color: #E8E8E8;
#         margin-bottom: 0px;
#     }
#     .invisbible-line {
#         height: 60px;
#     }
#     .section-header {
#         writing-mode: vertical-rl;
#         transform: rotate(180deg);
#         font-size: 24px;
#         margin-right: 20px;
#         margin-top: 25%;
#         margin-bottom: 25%;
#     }
#     .header-button {
#         display: block;
#         margin: 0 auto;
#     }
#     </style>
#     """, unsafe_allow_html=True)
#
# # Page heading
# st.title("Paper Review Summary")
#
# # Return to upload page button
# if st.button('Return to upload page'):
#     st.write("Redirecting to the upload page...")
#
# # Attitude Roots
# with stylable_container(
#     key="container_with_border",
#     css_styles="""
#         {
#             border-radius: 0.5rem;
#             padding: calc(1em - 1px);
#             background-color: white;
#             display: flex;
#             height: 100%;
#         }
#         """,
# ):
#     with st.container():
#         # Create a two-column layout
#         md = col1, col2 = st.columns([1, 9])
#
#         with col1:
#             # Vertical heading on the left
#             st.markdown('<div class="section-header">ATTITUDE ROOT</div>', unsafe_allow_html=True)
#
#         with col2:
#             for index, row in attitude_roots.iterrows():
#
#                 # Show Progress Bar with describtion
#                 col21, col22 = st.columns([5, 1])  # Adjust column widths
#                 with col21:
#                     st.markdown(f"<h4 style='font-size:18px; margin: 0px; padding: 0px;'>{row['Attitude root + Theme']}</h4>", unsafe_allow_html=True)
#                 with col22:
#                     st.markdown(f"<h4 style='font-size:18px; margin: 0px; padding: 0px;'>expand</h4>", unsafe_allow_html=True)
#                 st.progress(eval(row['Frequency']))
#
#                 # Description
#                 st.markdown(f"<div style='font-size:12px; margin-top: 0px; padding: 0px; margin-bottom: 5px;'>{row['Description']}</div>", unsafe_allow_html=True)
#
#                 st.markdown(f'<div class="content-box">Content of box </div>', unsafe_allow_html=True)
#                 st.markdown(f'<div class="invisbible-line">  </div>', unsafe_allow_html=True)
#
#
