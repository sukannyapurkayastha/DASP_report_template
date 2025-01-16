import pytest
from unittest.mock import call
from frontend.modules.shared_methods import (
    get_colour_palette,
    select_color_attitude_or_request,
    show_header_with_progress,
    show_reviewers_attitude_comments,
    draw_progress_bar,
    show_comments,
    use_default_container,
)


def test_get_colour_palette():
    palette = get_colour_palette()
    assert isinstance(palette, dict)
    assert 'bad_m4' in palette
    assert palette['bad_m4'] == '#e6550d'
    assert 'good_4' in palette
    assert palette['good_4'] == '#756cb2'


@pytest.mark.parametrize(
    "fraction, expected_color",
    [
        (0.0, '#756cb2'),  # 0.0 < 0.125 -> 'good_4'
        (0.1, '#756cb2'),  # 0.1 < 0.125 -> 'good_4'
        (0.2, '#9e9ac8'),  # 0.2 < 0.250 -> 'good_3'
        (0.3, '#bcbddc'),  # 0.3 < 0.375 -> 'good_2'
        (0.4, '#dadaeb'),  # 0.4 < 0.500 -> 'good_1'
        (0.5, '#fdd0a2'),  # 0.5 < 0.625 -> 'bad_m1'
        (0.7, '#fdae6b'),  # 0.7 < 0.750 -> 'bad_m2'
        (0.8, '#fd8d3c'),  # 0.8 < 0.875 -> 'bad_m3'
        (0.9, '#e6550d'),  # 0.9 > 0.875 -> 'bad_m4' (else statement)
    ],
)
def test_select_color_attitude_or_request(fraction, expected_color):
    color = select_color_attitude_or_request(fraction)
    assert color == expected_color


def test_draw_progress_bar(mocker):
    """
    Tests whether draw_progress_bar performs the expected st.markdown calls.
    We only check here that st.markdown is called and, if necessary, whether the HTML string
    is contained correctly.
    """
    mock_markdown = mocker.patch("streamlit.markdown")

    draw_progress_bar("#ff0000", 50)

    # We expect one call (had the problem that it was called multiple times).
    assert mock_markdown.call_count == 1
    args, kwargs = mock_markdown.call_args
    assert "background-color: #ff0000" in args[0]
    assert "width: 50%" in args[0]
    assert kwargs["unsafe_allow_html"] is True


def test_show_header_with_progress(mocker):
    """
    Tests whether show_header_with_progress draws the progress bar and displays the description if necessary.
    """
    mock_draw_bar = mocker.patch("frontend.modules.shared_methods.draw_progress_bar")
    mock_markdown = mocker.patch("streamlit.markdown")

    # row-Format: [Text, fraction, text]
    row = ["Test", 0.3, "Description text"]
    show_header_with_progress(row, desc=True)

    # fraction = 0.3 => color=select_color_attitude_or_request(0.3) => good_2 => "#bcbddc"
    # percentage = 30
    mock_draw_bar.assert_called_once_with('#bcbddc', 30)

    mock_markdown.assert_any_call(
        "<h4 style='font-size:12px; margin: 0px; padding: 0px; text-align: right;'>Description text</h4>",
        unsafe_allow_html=True
    )


@pytest.mark.parametrize(
    "desc", [True, False]
)
def test_show_header_with_progress_no_desc(mocker, desc):
    """
    Tests that an “invisible-line-minor” div is marked down if desc=False.
    """
    mock_markdown = mocker.patch("streamlit.markdown")

    row = ["Test", 0.3, "Description text"]
    show_header_with_progress(row, desc=desc)

    if desc:
        mock_markdown.assert_any_call(
            "<h4 style='font-size:12px; margin: 0px; padding: 0px; text-align: right;'>Description text</h4>",
            unsafe_allow_html=True
        )
    else:
        # invisbible-line-minor
        mock_markdown.assert_any_call(
            '<div class="invisbible-line-minor">  </div>',
            unsafe_allow_html=True
        )


def test_show_reviewers_attitude_comments(mocker):
    """
    Tests whether show_reviewers_attitude_comments calls each comment line via st.markdown.
    """
    mock_markdown = mocker.patch("streamlit.markdown")

    # row[i], row[i+1], ... => Strings
    row = ["Ignore", "Ignore", "Ignore", "Comment 1", "Comment 2", 123, "Not a comment", "Comment 3"]
    # The loop runs as long as row[i] is a str
    # => “Comment 1”, “Comment 2” and “Not a comment” are strings
    # Attention: 123 interrupts the loop because it is not a str!
    show_reviewers_attitude_comments(row)

    # We expect two calls (for “Comment 1” and “Comment 2”),
    # “Not a comment” is NOT reached because the loop ends as soon as we hit int 123.
    calls = [
        call('<div class="content-box">Comment 1</div>', unsafe_allow_html=True),
        call('<div class="content-box"> </div>', unsafe_allow_html=True),
        call('<div class="content-box">Comment 2</div>', unsafe_allow_html=True),
        call('<div class="content-box"> </div>', unsafe_allow_html=True),
    ]
    mock_markdown.assert_has_calls(calls, any_order=False)


def test_show_comments(mocker):
    """
    Tests show_comments, which makes st.markdown calls within a stylable_container.
    We mock stylable_container and st.markdown here.
    """
    mock_markdown = mocker.patch("streamlit.markdown")
    mock_stylable = mocker.patch("frontend.modules.shared_methods.stylable_container")

    row = {
        "Comments": [
            ["Reviewer 1", ["Nice paper!", "Please more examples."]],
            ["Reviewer 2", ["Well explained."]]
        ]
    }

    show_comments(row)

    # We expect that stylable_container is called once for each reviewer
    assert mock_stylable.call_count == 2

    # We check if the comments are in the markdown
    all_markdown_calls = "".join(str(c) for c in mock_markdown.call_args_list)
    assert "Reviewer 1" in all_markdown_calls
    assert "Nice paper!" in all_markdown_calls
    assert "Reviewer 2" in all_markdown_calls


def test_use_default_container_no_argument(mocker):
    """
    Tests that the function use_default_container with None argument calls the inside_container() without parameters.
    """
    mock_stylable = mocker.patch("frontend.modules.shared_methods.stylable_container")
    inside_mock = mocker.Mock()

    use_default_container(inside_mock, argument=None)

    assert mock_stylable.call_count == 1
    # inside_mock should be called with no arguments
    inside_mock.assert_called_once_with()


def test_use_default_container_with_argument(mocker):
    """
    Tests that the inside_container(argument) function is called when an argument is specified.
    """
    mock_stylable = mocker.patch("frontend.modules.shared_methods.stylable_container")
    inside_mock = mocker.Mock()

    use_default_container(inside_mock, argument="Test")

    assert mock_stylable.call_count == 1
    # inside_mock should be called with "Test" as specified above
    inside_mock.assert_called_once_with("Test")
