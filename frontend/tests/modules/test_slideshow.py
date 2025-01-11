import pytest
from unittest.mock import patch, MagicMock
import streamlit as st

# Suppose your code is in a file named `my_app.py`. Adjust if needed:
# from my_app import StreamlitSlideshow
from frontend.modules.slideshow import StreamlitSlideshow

import logging

# Supress streamlit warnings for using session_state even though there is no active streamlit running
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.state.session_state_proxy").setLevel(logging.ERROR)


@pytest.fixture
def sample_slides():
    """
    Returns a tuple (containers, slide_titles).
    Each container is just a simple function that we can patch later if needed.
    """
    def slide1():
        pass

    def slide2():
        pass

    containers = [slide1, slide2]
    titles = ["Slide 1 Title", "Slide 2 Title"]
    return containers, titles


def test_initialization(mocker, sample_slides):
    """
    Tests that the Slideshow initializes session_state['current_slide'] to 0 if not present.
    """
    mocker.patch.dict('streamlit.session_state', {}, clear=True)

    containers, titles = sample_slides
    slideshow = StreamlitSlideshow(containers, titles)

    # "current_slide" should be 0
    assert "current_slide" in st.session_state
    assert st.session_state["current_slide"] == 0


def test_go_to_next_slide(mocker, sample_slides):
    """
    Tests advancing the slide, including wrap-around when reaching the end.
    """
    mocker.patch.dict('streamlit.session_state', {"current_slide": 0}, clear=True)
    containers, titles = sample_slides
    slideshow = StreamlitSlideshow(containers, titles)

    # We have 2 slides total, so going to next from 0 => 1
    slideshow.go_to_next_slide()
    assert st.session_state["current_slide"] == 1

    # Going next from 1 => wrap around to 0
    slideshow.go_to_next_slide()
    assert st.session_state["current_slide"] == 0


def test_go_to_previous_slide(mocker, sample_slides):
    """
    Tests going to the previous slide, including wrap-around when at the beginning.
    """
    mocker.patch.dict('streamlit.session_state', {"current_slide": 0}, clear=True)
    containers, titles = sample_slides
    slideshow = StreamlitSlideshow(containers, titles)

    # current_slide=0 => going previous wraps around to the last slide
    slideshow.go_to_previous_slide()
    assert st.session_state["current_slide"] == len(containers) - 1

    # from last slide => previous => second-to-last, etc.
    slideshow.go_to_previous_slide()
    assert st.session_state["current_slide"] == len(containers) - 2


def test_render_slide_in_range(mocker, sample_slides):
    """
    Tests rendering a valid slide: ensures st.columns, st.button, and container are called.
    """
    mocker.patch.dict('streamlit.session_state', {"current_slide": 0}, clear=True)
    mock_columns = mocker.patch("frontend.modules.slideshow.st.columns")
    mock_button = mocker.patch("frontend.modules.slideshow.st.button")
    mock_markdown = mocker.patch("frontend.modules.slideshow.st.markdown")

    containers, titles = sample_slides
    slideshow = StreamlitSlideshow(containers, titles)

    # We'll track if container was called
    mock_container_fn = mocker.Mock()
    containers[0] = mock_container_fn

    # columns(...) returns mock columns that we can iterate
    col_left = MagicMock()
    col_center = MagicMock()
    col_right = MagicMock()
    mock_columns.return_value = [col_left, col_center, col_right]

    slideshow.render_slide()

    # st.columns should be called with [0.6, 8, 0.6]
    mock_columns.assert_called_once_with([0.6, 8, 0.6])

    # We expect 2 button calls (< and >) plus 1 markdown for the heading
    assert mock_button.call_count == 2
    assert mock_markdown.call_count == 1
    mock_container_fn.assert_called_once()

    args, kwargs = mock_markdown.call_args
    assert titles[0] in args[0]  # "Slide 1 Title"


def test_render_slide_out_of_range(mocker, sample_slides):
    """
    If current_slide is out of range, we expect st.error to be called.
    """
    mocker.patch.dict('streamlit.session_state', {"current_slide": 999}, clear=True)
    mock_error = mocker.patch("frontend.modules.slideshow.st.error")

    containers, titles = sample_slides
    slideshow = StreamlitSlideshow(containers, titles)
    slideshow.render_slide()

    mock_error.assert_called_once()
    args, kwargs = mock_error.call_args
    assert "Slide index out of range" in args[0]


def test_show(mocker, sample_slides):
    """
    Tests the top-level show() method, which just calls render_slide().
    """
    mock_render_slide = mocker.patch.object(StreamlitSlideshow, 'render_slide')
    containers, titles = sample_slides
    slideshow = StreamlitSlideshow(containers, titles)

    slideshow.show()
    mock_render_slide.assert_called_once()
