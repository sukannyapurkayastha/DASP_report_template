import streamlit as st
from streamlit_extras.stylable_container import stylable_container
import os
import pandas as pd
import matplotlib.pyplot as plt

#%% global variables

# Colors

red = '#ef2000'
orange = '#ff8220'
yellow = '#ffdc00'
grey = '#e0e0e0'
green_good = '#5aca00'
green_very_good = '#00b500'


# Import data

try:
    overview = pd.read_csv(os.path.join("dummy_data", "dummy_overview.csv"), sep=";", encoding="utf-8")
    attitude_roots = pd.read_csv(os.path.join("dummy_data", "dummy_attitude_roots.csv"), sep=";", encoding="utf-8")
    request_information = pd.read_csv(os.path.join("dummy_data", "dummy_request_information.csv"), sep=";", encoding="utf-8")
    summary = pd.read_csv(os.path.join("dummy_data", "dummy_summary.csv"), sep=";", encoding="utf-8")
except:
    FileNotFoundError("Some files have not been found.")

#%% Return to upload page logic
    
def switch_to_landing_page():
    st.session_state["current_page"] = "landing_page"
    
#TODO: we may want to allow the insertion of new URL immediatly without return to landing page.
def open_submenu_for_further_analysis():
    return None


#%% Create Mainpage

def main_page(custom_css):

    #%%% Set the page configuration
    st.set_page_config(
        page_title="Paper Review Summary",
        page_icon=os.path.join("logo.png")
        )
    
    # Apply custom CSS Styles
    
    st.markdown(custom_css, unsafe_allow_html=True)
    
    
    #%%% Special methods
    
    #%%%%% Shared methods
    
    def select_color_attitude_or_request(fraction):
        if fraction < 0.34:
            return yellow
        elif fraction < 0.67:
            return orange
        elif fraction <= 1:
            return red
    
    def draw_progress_bar(color, percentage):
        # Custom progress bar with percentage text centered
        progress_html = f"""
        <div style="position: relative; height: 24px; width: 100%; background-color: {grey}; border-radius: 5px;">
            <div style="width: {percentage}%; background-color: {color}; height: 100%; border-radius: 5px;">
            </div>
            <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); 
                         font-weight: bold; color: black; z-index: 1;">
                {percentage}%
            </div>
        </div>
        """
        st.markdown(progress_html, unsafe_allow_html=True)
    
    def write_attitude_root_or_request_heading(heading_text):
        st.markdown(f"<h4 style='font-size:18px; margin: 0px; padding: 0px;'>{heading_text}</h4>", unsafe_allow_html=True)
    
    #%%%%%Overview methods
    
    def select_color_and_text_overview(fraction):
        if fraction < 25:
            return red, 'poor'
        elif fraction < 50:
            return yellow, 'fair'
        elif fraction < 75:
            return green_good, 'good'
        elif fraction <= 100:
            return green_very_good, 'excellent'
        
    def get_score_text_overall_rating(score_percent):
        if score_percent > 9:
            return 'Strong accept, should be highlighted at the conference.'
        elif score_percent > 7:
            return 'Accept, good paper.'
        elif score_percent > 5:
            return 'Marginally above the acceptance threshold.'
        elif score_percent > 4:
            return 'Marginally below the acceptance threshold.'
        elif score_percent > 2:
            return 'Reject, not good enough.'
        elif score_percent > 0:
            return 'Strong reject.'
        else:
            Exception("Unexpected score!")
    
    def draw_attribute(column_name):
        if column == 'Overall Rating:':
            score_percent = overview[column_name].loc[0]*10
            color, score_text = select_color_and_text_overview(score_percent)
            st.markdown(f"<h4 style='font-size:18px; margin: 0px; padding: 0px; text-align: left;'>{get_score_text_overall_rating(score_percent)}</h4>", unsafe_allow_html=True)
            st.markdown('<div class="invisbible-line-minor">  </div>', unsafe_allow_html=True)
            draw_progress_bar(color, score_percent)
            st.markdown('<div class="invisbible-line-small">  </div>', unsafe_allow_html=True)
            
    
        else:
            score_percent = (overview[column_name].loc[0]/4)*100
            color, score_text = select_color_and_text_overview(score_percent)
            draw_progress_bar(color, score_percent)
            st.markdown(f"<h4 style='font-size:16px; margin: 0px; padding: 0px; text-align: right;'>{score_text}</h4>", unsafe_allow_html=True)
        
    
    #%%%%% Attitude Root methods
    
    def show_attitude_root_header(row):
        fraction = eval(row[1])
        percentage = int(fraction * 100)  # Convert fraction to percentage
        color = select_color_attitude_or_request(fraction)
        write_attitude_root_or_request_heading(row[0]) # writes attitude root heading
        
        draw_progress_bar(color, percentage) # draws the progressbar
    
        st.write(row[2]) #Writes the description
     
        
    def show_reviewers_attitude_comments(row):
        i = 3
        while i < len(row) and isinstance(row[i], str):
            st.markdown(f'<div class="content-box">{row[i]}</div>', unsafe_allow_html=True)
            st.markdown('<div class="content-box"> </div>', unsafe_allow_html=True)
            i += 1  
    
    
    #%%%%% Request methods
    
    def draw_circle(row):
        fraction = eval(row[1])
        sizes = [1 - fraction, fraction]  # Percentages or values
        colors = [grey, select_color_attitude_or_request(fraction)] #grey + selected color (red/orange/yellow)
            
        # Create the pie chart
        fig, ax = plt.subplots()
        ax.pie(
            sizes,
            #explode = (0, 0.1),  
            colors=colors, 
            #autopct='%1.1f%%', 
            startangle=90,  # Rotate so the first slice starts at 12 o'clock
            wedgeprops={'width': 0.4, 'edgecolor': 'w'}  # Create the "donut" effect
        )
        
        # Add a white circle in the center
        center_circle = plt.Circle((0, 0), 0.75, color='white', fc='white', linewidth=0)
        ax.add_artist(center_circle)
        
        ax.text(
        0, 0,  # Coordinates for the center
        str(int(eval(row[1]) * 100)) + '%',  # The text to display
        ha='center',  # Horizontal alignment
        va='center',  # Vertical alignment
        fontsize=26,  # Font size
        weight = 'bold',
        color='black'  # Font color
        )
        
        # Equal aspect ratio ensures the pie is drawn as a circle
        ax.axis('equal')
        
        # Show the plot
        st.pyplot(fig)
        
            
    def show_reviewers_request_comments(row):
        i = 2
        while i < len(row) and isinstance(row[i], str):
            st.markdown(f'<div class="content-box">{row[i]}</div>', unsafe_allow_html=True)
            st.markdown('<div class="content-box"> </div>', unsafe_allow_html=True)
            i += 1
    
    
    #%%% Page-Header
    
    st.title("Paper Review Summary")

    # Create returnbutton
    st.button("Return to upload page", on_click=switch_to_landing_page)
    
    
    #%%%Overview
    
    with stylable_container(
        key="container_with_border",
        css_styles="""
            {
                border-radius: 0.5rem;
                padding: calc(1em - 1px);
                background-color: white;
                display: flex;
                height: 100%;
            }
            """,
    ):
        with st.container():
            # Create a two-column layout
            col1, col2 = st.columns([1, 9])
    
            with col1:
                # Vertical heading on the left
                st.markdown('<div class="section-header">OVERVIEW</div>', unsafe_allow_html=True)
                
    
            with col2:
                if overview.empty:
                    write_attitude_root_or_request_heading("No general Information was found.")
                else:
                    st.markdown('<div class="invisbible-line-small">  </div>', unsafe_allow_html=True)
                    for column in overview:
                        with st.container():
                            col1, col2 = st.columns([2.1,8])
                            with col1:
                                write_attitude_root_or_request_heading(column)
                            
                            with col2:
                                draw_attribute(column)
                                st.markdown('<div class="invisbible-line-small">  </div>', unsafe_allow_html=True)
    
    
    #%%% Attitude Roots
    
    with stylable_container(
        key="container_with_border",
        css_styles="""
            {
                border-radius: 0.5rem;
                padding: calc(1em - 1px);
                background-color: white;
                display: flex;
                height: 100%;
            }
            """,
    ):
        with st.container():
            # Create a two-column layout
            col1, col2 = st.columns([1, 9])
    
            with col1:
                # Vertical heading on the left
                st.markdown('<div class="section-header">ATTITUDE ROOT</div>', unsafe_allow_html=True)
                
    
            with col2:
                if attitude_roots.empty:
                    write_attitude_root_or_request_heading("No Attitude Roots were found.")
                else:
                    for index, row in attitude_roots.iterrows():
                        show_attitude_root_header(row)
                        with st.expander("Comments"):
                            show_reviewers_attitude_comments(row)
                        st.markdown('<div class="invisbible-line-big">  </div>', unsafe_allow_html=True)
                        
    
    
    #%%% Request Information
    
    with stylable_container(
        key="container_with_border",
        css_styles="""
            {
                border-radius: 0.5rem;
                padding: calc(1em - 1px);
                background-color: white;
                display: flex;
                height: 100%;
            }
            """,
    ):
                        
        with st.container():
            # Create a two-column layout
            col1, col2 = st.columns([1, 9])
    
            with col1:
                # Vertical heading on the left
                st.markdown('<div class="section-header">REQUEST INFORMATION</div>', unsafe_allow_html=True)
                
                
            with col2:
                if request_information.empty:
                    write_attitude_root_or_request_heading("No Requests were found.")
                else:
                    for index, row in request_information.iterrows():
                        with st.container():
                            write_attitude_root_or_request_heading(row[0])
                            col3, col4 = st.columns([3,6])
                            with col3:
                                draw_circle(row)
                                    
                            with col4:
                                with st.expander("Comments"):
                                    show_reviewers_request_comments(row)
                                st.markdown('<div class="invisbible-line-big">  </div>', unsafe_allow_html=True)
    
    
    #%%% Summary
    
    with stylable_container(
        key="container_with_border",
        css_styles="""
            {
                border-radius: 0.5rem;
                padding: calc(1em - 1px);
                background-color: white;
                display: flex;
                height: 100%;
            }
            """,
    ):                    
        with st.container():
            # Create a two-column layout
            col1, col2 = st.columns([1, 9])
    
            with col1:
                # Vertical heading on the left
                st.markdown('<div class="section-header">SUMMARY</div>', unsafe_allow_html=True)
    
            with col2:
                if summary.empty:
                    write_attitude_root_or_request_heading("No summary was found.")
                else:
                    for index, row in summary.iterrows():
                        st.write(row[0]) # write summary
