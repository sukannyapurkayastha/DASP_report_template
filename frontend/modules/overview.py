import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from .shared_methods import get_colour_palette, grey
from .shared_methods import use_default_container
from streamlit_extras.stylable_container import stylable_container

def show_overview(overview_data):
    # Helper methods
    def select_color_and_text_overview(fraction):
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
        if category == 'Rating':
            fraction = score / 10
        elif category in ['Soundness', 'Presentation', 'Contribution']:
            fraction = score / 4
        else:
            fraction = score  # Oder eine andere Standardbehandlung
        return min(max(fraction, 0), 1)  # Sicherstellen, dass fraction zwischen 0 und 1 liegt

    def draw_donut_chart(fraction, score_text):
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
        score = float(row['Avg_Score'])
        category = row['Category']
        fraction = calculate_fraction(score, category)
        draw_donut_chart(fraction, score_text=score)

    def draw_individual_circle(score_value, category, author):
        try:
            score_value = float(score_value)
        except (ValueError, TypeError):
            st.markdown(f"Invalid score value for {author}")
            return

        fraction = calculate_fraction(score_value, category)
        draw_donut_chart(fraction, score_text=score_value)
        st.markdown(f"<p style='text-align:center;'>{author}</p>", unsafe_allow_html=True)

    def show_overview_individual(row_data):
        category = row_data['Category']
        individual_scores_list = row_data['Individual_scores']
        if not individual_scores_list or individual_scores_list == 'None':
            st.markdown(f"No individual scores available for {category}")
        else:
            num_scores = len(individual_scores_list)
            st_columns = st.columns(num_scores)
            for idx, (author_score, st_col) in enumerate(zip(individual_scores_list, st_columns)):
                author, score_value = author_score
                with st_col:
                    if score_value is None or score_value == 'None':
                        st.markdown(f"No data available for {author}")
                    else:
                        draw_individual_circle(score_value, category, author)

    # Actual overview generation
    with st.container():
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
