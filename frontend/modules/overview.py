# overview.py

"""
Overview Module

This module provides functions to display an overview of paper reviews within the
Paper Review Aggregator application. It includes utilities for selecting colors
based on review fractions, calculating fractions from scores, drawing donut charts,
and rendering individual and overall overview sections. The module leverages
Streamlit for interactive web components and integrates with shared methods for
consistent styling and functionality.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from .shared_methods import get_colour_palette, grey
from .shared_methods import use_default_container
from streamlit_extras.stylable_container import stylable_container
import math


# Helper methods
def select_color_and_text_overview(fraction):
    """
    Select an appropriate color based on the provided fraction for the overview section.
    
    This function maps a numerical fraction to a specific color from the predefined
    color palette, indicating different levels of classification in the overview.
    
    Args:
        fraction (float): A numerical value representing the fraction or proportion.
    
    Returns:
        str: A hex color code corresponding to the classification level.
    """
    palette = get_colour_palette()
    if fraction < 0.125:
        return palette['bad_m4']
    elif fraction < 0.25:
        return palette['bad_m3']
    elif fraction < 0.375:
        return palette['bad_m2']
    elif fraction < 0.5:
        return palette['bad_m1']
    elif fraction < 0.625:
        return palette['good_1']
    elif fraction < 0.75:
        return palette['good_2']
    elif fraction < 0.875:
        return palette['good_3']
    else:
        return palette['good_4']


def calculate_fraction(score, category):
    """
    Calculate the fraction based on the score and category.
    
    This function normalizes the score to a fraction between 0 and 1 depending on
    the category of the score. For 'Rating', the score is divided by 10, and for
    'Soundness', 'Presentation', and 'Contribution', the score is divided by 4.
    Any other category retains the original score value.
    
    Args:
        score (float): The score value to be normalized.
        category (str): The category of the score (e.g., 'Rating', 'Soundness').
    
    Returns:
        float: A normalized fraction between 0 and 1.
    """
    if category == 'Rating':
        fraction = score / 10
    elif category in ['Soundness', 'Presentation', 'Contribution']:
        fraction = score / 4
    else:
        fraction = score  # Or another default handling
    return min(max(fraction, 0), 1)  # Ensure fraction is between 0 and 1


def draw_donut_chart(fraction, score_text):
    """
    Draw a donut chart representing the fraction with a central score text.
    
    This function creates a donut-shaped pie chart using Matplotlib, where the filled
    portion corresponds to the provided fraction. The central area displays the score
    text. The chart is then rendered in the Streamlit application.
    
    Args:
        fraction (float): The fraction to be represented in the donut chart.
        score_text (str): The text to be displayed at the center of the donut chart.
    """
    sizes = [1 - fraction, fraction]
    signal_color = select_color_and_text_overview(fraction)
    colors = [grey, signal_color]

    fig, ax = plt.subplots()
    ax.pie(
        sizes,
        colors=colors,
        startangle=90,
        wedgeprops={'width': 0.4, 'edgecolor': 'w'}
    )

    center_circle = plt.Circle((0, 0), 0.75, color='white', fc='white', linewidth=0)
    ax.add_artist(center_circle)

    ax.text(
        0, 0,
        f"{score_text}",
        ha='center',
        va='center',
        fontsize=35,
        weight='bold',
        color='black'
    )

    ax.axis('equal')
    st.pyplot(fig)


def draw_circle(row):
    """
    Draw a donut chart for a specific category based on the row data.
    
    This function extracts the average score and category from the provided row,
    calculates the corresponding fraction, and renders a donut chart with the score.
    
    Args:
        row (pd.Series): A row from the DataFrame containing 'Avg_Score' and 'Category'.
    """
    score = float(row['Avg_Score'])
    category = row['Category']
    fraction = calculate_fraction(score, category)
    draw_donut_chart(fraction, score_text=score)


def draw_individual_circle(score_value, category, author):
    """
    Draw an individual donut chart for a specific author's score.
    
    This function validates the score value, calculates the corresponding fraction,
    and renders a donut chart with the score. It also displays the author's name
    below the chart.
    
    Args:
        score_value (str or float): The score value provided by the author.
        category (str): The category of the score (e.g., 'Rating', 'Soundness').
        author (str): The name of the author providing the score.
    """
    try:
        score_value = float(score_value)
    except (ValueError, TypeError):
        st.markdown(f"Invalid score value for {author}")
        return

    fraction = calculate_fraction(score_value, category)
    draw_donut_chart(fraction, score_text=score_value)
    st.markdown(f"<p style='text-align:center;'>{author}</p>", unsafe_allow_html=True)


def show_overview_individual(row_data):
    """
    Display individual scores within the overview section.
    
    This function organizes individual scores into a grid layout, determining the
    number of rows and columns based on the number of scores. Each score is rendered
    as an individual donut chart with the author's name.
    
    Args:
        row_data (pd.Series): A row from the DataFrame containing 'Category' and 'Individual_scores'.
    """
    category = row_data['Category']
    individual_scores_list = row_data['Individual_scores']
    if not individual_scores_list or individual_scores_list == 'None':
        st.markdown(f"No individual scores available for {category}")
    else:
        num_scores = len(individual_scores_list)
        num_columns = 4
        num_rows = math.ceil(num_scores / num_columns)

        for row in range(num_rows):
            st_columns = st.columns(num_columns)
            for idx in range(num_columns):
                index = row * num_columns + idx
                if index < num_scores:
                    author, score_value = individual_scores_list[index]
                    with st_columns[idx]:
                        if score_value is None or score_value == 'None':
                            st.markdown(f"No data available for {author}")
                        else:
                            draw_individual_circle(score_value, category, author)


def show_overview(overview_data):
    """
    Display the overview section with aggregated review data.
    
    This function renders the main overview section, including donut charts for each
    category and a slideshow containing detailed information on attitude roots,
    request information, and summary. It handles cases where data might be missing
    by displaying appropriate warnings and loading dummy data if necessary.
    
    Args:
        overview_data (pd.DataFrame): A DataFrame containing aggregated overview information.
    """
    with st.container():
        st.title("Paper Review Aggregation")
        st.markdown('<div class="invisbible-line-minor">  </div>', unsafe_allow_html=True)

        if overview_data.empty:
            st.markdown(
                "<h4 style='font-size:16px; margin: 0px; padding: 0px;'>No general information was found.</h4>",
                unsafe_allow_html=True
            )
        else:
            num_rows = len(overview_data)
            st_columns = st.columns(num_rows)
            for idx, (index, row_data), st_col in zip(range(num_rows), overview_data.iterrows(), st_columns):
                with st_col:
                    draw_circle(row_data)
                    with st.popover(f"{row_data['Category']}"):
                        show_overview_individual(row_data)
