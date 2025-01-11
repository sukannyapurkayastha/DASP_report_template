import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

from frontend.modules.overview import (
    show_overview,
    calculate_fraction,
    select_color_and_text_overview,
    get_colour_palette,
    draw_donut_chart
)


@pytest.fixture
def sample_overview_data():
    # Provide a small DataFrame with the structure that `overview_data` expects
    data = {
        "Category": ["Rating", "Soundness", "Presentation"],
        "Avg_Score": [9.5, 2.5, 3.5],
        "Individual_scores": [
            [("Reviewer A", 9.0), ("Reviewer B", 10.0)],
            [("Reviewer A", 2.0), ("Reviewer B", 3.0)],
            [("Reviewer A", 4.0), ("Reviewer B", 3.0)]
        ],
    }
    return pd.DataFrame(data)


@pytest.mark.parametrize(
    "score,category,expected_fraction",
    [
        (10, "Rating", 1.0),  # 10 out of 10 => fraction=1.0 (capped)
        (9, "Rating", 0.9),
        (2, "Soundness", 0.5),  # 2 out of 4 => fraction=0.5
        (10, "Soundness", 1.0),  # 10 out of 4 => fraction=1.0 (capped)
        (-1, "Presentation", 0.0),  # negative => fraction=0.0 (min)
        (2, "Presentation", 0.5),
    ],
)
def test_calculate_fraction_logic(score, category, expected_fraction):
    """
    Tests calculation of fractions with valid and invalid (impossible) inputs.
    """
    fraction = calculate_fraction(score, category)
    assert fraction == pytest.approx(expected_fraction, 0.001)


@pytest.mark.parametrize(
    "fraction,expected_color_key",
    [
        (0.0, "bad_m4"),  # 0.0 < 0.125 => 'bad_m4'
        (0.1, "bad_m4"),  # 0.1 < 0.125 => 'bad_m4'
        (0.24, "bad_m3"),  # 0.24 < 0.25 => 'bad_m3'
        (0.3, "bad_m2"),  # 0.3 < 0.375 => 'bad_m2'
        (0.49, "bad_m1"),  # 0.49 < 0.5 => 'bad_m1'
        (0.6, "good_1"),  # 0.6 < 0.625 => 'good_1'
        (0.7, "good_2"),  # 0.7 < 0.75 => 'good_2'
        (0.8, "good_3"),  # 0.8 < 0.875 => 'good_3'
        (0.9, "good_4"),  # 0.9 >= 0.875 => 'good_4'
    ],
)
def test_select_color_and_text_overview_logic(fraction, expected_color_key):
    color = select_color_and_text_overview(fraction)
    palette = get_colour_palette()
    assert color == palette[expected_color_key]


def test_draw_donut_chart(mocker):
    """
    Verifies that `draw_donut_chart` calls `st.pyplot` once and that matplotlib pie logic doesn't break.
    """
    mock_pyplot = mocker.patch("frontend.modules.overview.st.pyplot")
    mock_plt_subplots = mocker.patch("frontend.modules.overview.plt.subplots")
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plt_subplots.return_value = (mock_fig, mock_ax)

    # We do not check the actual figure content, just that st.pyplot is called.
    draw_donut_chart(fraction=0.5, score_text="5.0")

    assert mock_pyplot.call_count == 1
    assert mock_plt_subplots.call_count == 1


def test_show_overview_empty(mocker):
    """
    Tests show_overview with an empty DataFrame.
    """
    mock_markdown = mocker.patch("frontend.modules.overview.st.markdown")
    mock_title = mocker.patch("frontend.modules.overview.st.title")
    mock_container = mocker.patch("frontend.modules.overview.st.container")  # Need to mock it for with st.container()

    empty_df = pd.DataFrame(columns=["Category", "Avg_Score", "Individual_scores"])
    show_overview(empty_df)

    # Expect st.title to be called
    mock_title.assert_called_once_with("Paper Review Aggregation")
    # The code checks `if overview_data.empty: ...`
    calls_texts = " ".join(str(call) for call in mock_markdown.call_args_list)
    assert "No general information" in calls_texts


def test_show_overview(mocker, sample_overview_data):
    """
    High-level test for show_overview.
    """
    mock_title = mocker.patch("frontend.modules.overview.st.title")
    mock_markdown = mocker.patch("frontend.modules.overview.st.markdown")
    mock_container = mocker.patch("frontend.modules.overview.st.container")
    mock_columns = mocker.patch("frontend.modules.overview.st.columns")
    mock_popover = mocker.patch("frontend.modules.overview.st.popover")
    mock_pyplot = mocker.patch("frontend.modules.overview.st.pyplot")

    # st.columns(...) returns a list of mock column contexts
    mock_col1, mock_col2, mock_col3 = MagicMock(), MagicMock(), MagicMock()
    mock_columns.return_value = [mock_col1, mock_col2, mock_col3]

    show_overview(sample_overview_data)

    mock_title.assert_called_once_with("Paper Review Aggregation")
    # We expect at least one st.markdown call for the "invisible line" or something
    assert mock_markdown.call_count >= 1

    # Check that columns were created (3 calls, one for each row + 1 one call for checking length)
    assert mock_columns.call_count == len(sample_overview_data.columns) + 1

    # Check that st.pyplot is called, meaning some chart was drawn
    # (Three donut chart per row in sample_overview_data => 9 times)
    assert mock_pyplot.call_count == len(sample_overview_data) * 3

    # We expect a popover for each row, so 3 times in this example
    assert mock_popover.call_count == len(sample_overview_data)
