import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from modules.shared_methods import red, orange, yellow, grey, green_good, green_very_good


def show_overview(overview_data):
    
    #%% Helper methods
    def select_color_and_text_overview(fraction):
        if fraction < 25:
            return red, 'poor'
        elif fraction < 50:
            return yellow, 'fair'
        elif fraction < 75:
            return green_good, 'good'
        elif fraction <= 100:
            return green_very_good, 'excellent'
        
    def draw_circle(column_name):
        #name = (overview_data[column_name].loc[:])
        #st.markdown(column_name)
        fraction = (overview_data[column_name].loc[0])
        sizes = [1 - fraction, fraction]  # Percentages or values
        
        signal_color, result_word = select_color_and_text_overview(fraction*100)
        colors = [grey, signal_color] #grey + selected color (red/orange/yellow)
            
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
        column_name, #row[1].split("/")[0] + '\n requests',  # The text to display
        ha='center',  # Horizontal alignment
        va='center',  # Vertical alignment
        fontsize=25,  # Font size
        weight = 'bold',
        color='black'  # Font color
        )
        
        # Equal aspect ratio ensures the pie is drawn as a circle
        ax.axis('equal')
        
        # Show the plot
        st.pyplot(fig)
        
    #%% actual overview generation
    with st.container():
        if overview_data.empty:
           st.markdown("<h4 style='font-size:16px; margin: 0px; padding: 0px;'>No general Information was found.</h4>", unsafe_allow_html=True)
        else:
            # Create a column for each categorie
            columns = list(overview_data)
            num_columns = len(columns)
            # Create Streamlit columns dynamically based on the number of attributes
            st_columns = st.columns(num_columns)
            for idx, (column_name, st_col) in enumerate(zip(columns, st_columns)):
                with st_col:
                    draw_circle(column_name)
                    
    