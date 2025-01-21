# shared_methods.py

"""
Shared Methods Module

This module provides shared utility functions and methods used across various
parts of the Paper Review Aggregator application. It includes functions for
color selection based on fractions, rendering progress bars, displaying comments,
and managing styled containers within Streamlit.
"""

import streamlit as st
from streamlit_extras.stylable_container import stylable_container

# Define color codes
red = '#ef2000'
orange = '#ff8220'
yellow = '#ffdc00'
grey = '#e0e0e0'
green_good = '#5aca00'
green_very_good = '#00b500'


def get_colour_palette():
    """
    Retrieve the color palette for attitude or request classification.
    
    This function returns a dictionary mapping classification levels to their
    corresponding color codes.
    
    Returns:
        dict: A dictionary containing color codes for different classification levels.
    """
    return {
        'bad_m4': '#e6550d',
        'bad_m3': '#fd8d3c',
        'bad_m2': '#fdae6b',
        'bad_m1': '#fdd0a2',
        'good_1': '#dadaeb',
        'good_2': '#bcbddc',
        'good_3': '#9e9ac8',
        'good_4': '#756cb2',
    }


def select_color_attitude_or_request(fraction):
    """
    Select an appropriate color based on the provided fraction.
    
    This function maps a numerical fraction to a specific color in the palette,
    indicating different levels of classification.
    
    Args:
        fraction (float): A numerical value representing the fraction or proportion.
    
    Returns:
        str: A hex color code corresponding to the classification level.
    """
    palette = get_colour_palette()
    if fraction < 0.125:
        return palette['good_4']
    elif fraction < 0.25:
        return palette['good_3']
    elif fraction < 0.375:
        return palette['good_2']
    elif fraction < 0.5:
        return palette['good_1']
    elif fraction < 0.625:
        return palette['bad_m1']
    elif fraction < 0.75:
        return palette['bad_m2']
    elif fraction < 0.875:
        return palette['bad_m3']
    else:
        return palette['bad_m4']

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


def show_reviewers_attitude_comments(row):
    """
    Display reviewers' comments related to attitude classifications.
    
    This function iterates through the comments associated with a particular
    attitude root and renders them within styled containers.
    
    Args:
        row (pd.Series): A row from the DataFrame containing attitude root information.
    """
    i = 3
    while i < len(row) and isinstance(row[i], str):
        st.markdown(f'<div class="content-box">{row[i]}</div>', unsafe_allow_html=True)
        st.markdown('<div class="content-box"> </div>', unsafe_allow_html=True)
        i += 1


def draw_progress_bar(color, percentage):
    """
    Render a custom progress bar with the specified color and percentage.
    
    This function creates an HTML-based progress bar and displays it within the
    Streamlit application.
    
    Args:
        color (str): The color code for the filled portion of the progress bar.
        percentage (int): The percentage (0-100) to fill the progress bar.
    """
    progress_html = f"""
    <div style="position: relative; height: 24px; width: 100%; background-color: {grey}; border-radius: 5px;">
        <div style="width: {percentage}%; background-color: {color}; height: 100%; border-radius: 5px;">
        </div>
    </div>
    """
    st.markdown(progress_html, unsafe_allow_html=True)


def show_comments(row):
    """
    Display comments associated with a specific classification.
    
    This function renders each comment under an expandable section, showing the
    author's name and their respective comments.
    
    Args:
        row (pd.Series): A row from the DataFrame containing comments information.
    """
    comments = row['Comments']
    for author_comments in comments:
        author = author_comments[0]
        comment_list = author_comments[1]

        with stylable_container(
                key="container_without_border",
                css_styles="""
                {
                    border-radius: 0.5rem;
                    padding: calc(1em - 1px);
                    display: flex;
                    height: 100%;
                    background-color: #F2F2F2;
                    padding-bottom: 8px;
                    background-color: #E8E8E8;
                }
                """,
        ):
            with st.container():
                st.markdown(f'<div class="content-text">{author}</div>', unsafe_allow_html=True)
                st.markdown('\n'.join(f'- {comment}' for comment in comment_list))


def use_default_container(inside_container, argument=None):
    """
    Render content within a styled container.
    
    This function wraps the provided content within a styled container, applying
    specific CSS styles for consistent appearance across the application.
    
    Args:
        inside_container (callable): A callable function that renders the desired content.
        argument (optional): An optional argument to pass to the `inside_container` function.
    
    Example:
        use_default_container(modules.overview.show_overview, overview_data)
    """
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
        if argument is None:
            # No argument was provided
            inside_container()
        else:
            # Argument was provided
            inside_container(argument)
