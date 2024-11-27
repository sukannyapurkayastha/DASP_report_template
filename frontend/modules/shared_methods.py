import streamlit as st
from streamlit_extras.stylable_container import stylable_container
import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pickle

red = '#ef2000'
orange = '#ff8220'
yellow = '#ffdc00'
grey = '#e0e0e0' 
green_good = '#5aca00'
green_very_good = '#00b500'


def select_color_attitude_or_request(fraction):
    if fraction < 0.2:
        return green_good
    if fraction < 0.34:
        return yellow
    elif fraction < 0.67:
        return orange
    elif fraction <= 1:
        return red


def show_header_with_progress(row, desc):
    try: 
        fraction = eval(row[1])  # todo: the fraction should be a number --> remove eval
    except:
        fraction = row[1]
    percentage = int(fraction*100)  # Convert fraction to percentage
    color = select_color_attitude_or_request(fraction)
    draw_progress_bar(color, percentage) # draws the progressbar
    if desc: 
        st.markdown(f"<h4 style='font-size:12px; margin: 0px; padding: 0px; text-align: right;'>{row[2]}</h4>", unsafe_allow_html=True)
    else:
        st.markdown('<div class="invisbible-line-minor">  </div>', unsafe_allow_html=True)

def show_reviewers_attitude_comments(row):
    i = 3
    while i < len(row) and isinstance(row[i], str):
        st.markdown(f'<div class="content-box">{row[i]}</div>', unsafe_allow_html=True)
        st.markdown('<div class="content-box"> </div>', unsafe_allow_html=True)
        i += 1  
                    
                    
def draw_progress_bar(color, percentage):
    progress_html = f"""
    <div style="position: relative; height: 24px; width: 100%; background-color: {grey}; border-radius: 5px;">
        <div style="width: {percentage}%; background-color: {color}; height: 100%; border-radius: 5px;">
        </div>
    </div>
    """
    st.markdown(progress_html, unsafe_allow_html=True)           


def show_comments(row):
    comments = row['Comments']
    for author_comments in comments:
        author = author_comments[0]
        comment_list = author_comments[1]
        
       # st.markdown(f'### Kommentare von {author}')
       
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
        