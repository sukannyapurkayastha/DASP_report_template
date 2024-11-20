import streamlit as st
from streamlit_extras.stylable_container import stylable_container


#%% Import data
import pandas as pd
import os
import matplotlib.pyplot as plt

attitude_roots = pd.read_csv(os.path.join("dummy_data", "dummy_attitude_roots.csv"), sep=";", encoding="utf-8")
request_information = pd.read_csv(os.path.join("dummy_data", "dummy_request_information.csv"), sep=";", encoding="utf-8")
summary = pd.read_csv(os.path.join("dummy_data", "dummy_summary.csv"), sep=";", encoding="utf-8")


#%% Special methods

def select_color(fraction):
    if fraction < 0.34:
        return "#FFFF3F" #gelb
    elif fraction < 0.67:
        return "#FFA546" #orange
    elif fraction <= 1:
        return "#FF2C37" #rot   

def draw_progress_bar(row):
    fraction = eval(row[1])
    percentage = int(fraction * 100)  # Convert fraction to percentage
    color = select_color(fraction)
    col21, col22 = st.columns([5, 1])  # Adjust column widths
    with col21:
        st.markdown(f"<h4 style='font-size:18px; margin: 0px; padding: 0px;'>{row['Attitude root + Theme']}</h4>", unsafe_allow_html=True)
    with col22:
        st.markdown("<h4 style='font-size:18px; margin: 0px; padding: 0px;'>expand</h4>", unsafe_allow_html=True)

    # Custom progress bar with percentage text centered
    progress_html = f"""
    <div style="position: relative; height: 24px; width: 100%; background-color: #e0e0e0; border-radius: 5px;">
        <div style="width: {percentage}%; background-color: {color}; height: 100%; border-radius: 5px;">
        </div>
        <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); 
                     font-weight: bold; color: black; z-index: 1;">
            {percentage}%
        </div>
    </div>
    """
    st.markdown(progress_html, unsafe_allow_html=True)


    #st.progress(fraction, text=row[0]) #Frequency + 'Attitude Root + Theme'
    st.write(row[2]) #Description
   
def show_reviewers_attitude_comments(row):
    i = 3
    while i < len(row) and isinstance(row[i], str):
        st.markdown(f'<div class="content-box">{row[i]}</div>', unsafe_allow_html=True)
        st.markdown('<div class="content-box"> </div>', unsafe_allow_html=True)
        i += 1  

def draw_circle(row):
    fraction = eval(row[1])
    sizes = [1 - fraction, fraction]  # Percentages or values
    colors = ['grey', select_color(fraction)]
        
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
    color='black'  # Font color
)
    
    # Equal aspect ratio ensures the pie is drawn as a circle
    ax.axis('equal')
    
    # Show the plot
    st.pyplot(fig)

def write_request(row):
        st.markdown(f"<h4 style='font-size:18px; margin: 0px; padding: 0px;'>{row[0]}</h4>", unsafe_allow_html=True)
        
def show_reviewers_request_comments(row):
    i = 2
    while i < len(row) and isinstance(row[i], str):
        st.markdown(f'<div class="content-box">{row[i]}</div>', unsafe_allow_html=True)
        st.markdown('<div class="content-box"> </div>', unsafe_allow_html=True)
        i += 1


#%% Create page



# Set the page configuration
st.set_page_config(page_title="Paper Review Summary")

# Apply custom CSS styles
st.markdown("""
    <style>
    body {
        background-color: #F2F2F2 !Important;
    }
    .stApp {
        background-color: #F2F2F2; /* For Streamlit app container */
    }
    .content-box {
        padding-left: 2px;
        margin-bottom: 0px;
        border-radius: 5px;
        background-color: #E8E8E8;
        margin-bottom: 2px;
    }
    .invisbible-line {
        height: 60px;
    }
    .section-header {
        writing-mode: vertical-rl;
        transform: rotate(180deg);
        font-size: 24px;
        margin-right: 20px;
        margin-top: 25%;
        margin-bottom: 25%;
        
    }
    </style>
    """, unsafe_allow_html=True)

# Page heading
st.title("Paper Review Summary")


#%% Navigation

# Return to upload page button
if st.button('Return to upload page'):
    st.write("Redirecting to the upload page...")


#%% Attitude Roots
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
        md = col1, col2 = st.columns([1, 9])

        with col1:
            # Vertical heading on the left
            st.markdown('<div class="section-header">ATTITUDE ROOT</div>', unsafe_allow_html=True)
            

        with col2:
            # Create 3 white boxes under each other
            for index, row in attitude_roots.iterrows():
                draw_progress_bar(row)
                if True: #TODO: if expand button is pressed
                    show_reviewers_attitude_comments(row)
                st.markdown('<div class="invisbible-line">  </div>', unsafe_allow_html=True)
                    


#%% Request Information
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
        md = col1, col2 = st.columns([1, 9])

        with col1:
            # Vertical heading on the left
            st.markdown('<div class="section-header">REQUEST</div>', unsafe_allow_html=True)
            
            
        with col2:
            for index, row in request_information.iterrows():
                with st.container():
                    col3, col4 = st.columns([3,6])
                    with col3:
                        draw_circle(row)
                            
                    with col4:
                        write_request(row)
                        if True: #TODO: if expand button is pressed
                            show_reviewers_request_comments(row)
                        st.markdown('<div class="invisbible-line">  </div>', unsafe_allow_html=True)


#%% Summary
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
        md = col1, col2 = st.columns([1, 9])

        with col1:
            # Vertical heading on the left
            st.markdown('<div class="section-header">SUMMARY</div>', unsafe_allow_html=True)

        with col2:
            for index, row in summary.iterrows():
                st.write(row[0]) # write summary
            

    
