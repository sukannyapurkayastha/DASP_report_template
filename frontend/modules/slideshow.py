# slideshow.py

"""
Slideshow Module

This module defines the `StreamlitSlideshow` class, which facilitates the creation and
management of a slideshow within a Streamlit application. It allows users to navigate
through different slides, each containing specific content, using "Previous" and "Next"
buttons. The slideshow maintains the current slide state using Streamlit's session state.
"""

import streamlit as st


# Define the Slideshow class
class StreamlitSlideshow:
    def __init__(self, containers, slide_titles):
        """
        Initialize the slideshow with a list of containers and their titles.
        
        Args:
            containers (list): A list of callable Streamlit containers.
            slide_titles (list): A list of titles for the slides.
        """
        self.containers = containers
        self.slide_titles = slide_titles

        # Initialize session state for current slide
        if "current_slide" not in st.session_state:
            st.session_state["current_slide"] = 0  # Start with the first slide

    def go_to_previous_slide(self):
        """
        Navigate to the previous slide.
        
        This method decrements the current slide index, wrapping around to the last slide
        if the current slide is the first one.
        """
        st.session_state["current_slide"] = (st.session_state["current_slide"] - 1) % len(self.containers)

    def go_to_next_slide(self):
        """
        Navigate to the next slide.
        
        This method increments the current slide index, wrapping around to the first slide
        if the current slide is the last one.
        """
        st.session_state["current_slide"] = (st.session_state["current_slide"] + 1) % len(self.containers)

    def render_slide(self):
        """
        Render the current slide based on the index stored in session state.
        
        This method displays the navigation buttons ("Previous" and "Next") and the content
        of the current slide by invoking the corresponding container function.
        
        Raises:
            StreamlitAPIException: If the slide index is out of the valid range.
        """
        slide_index = st.session_state["current_slide"]

        # Ensure the slide index is valid
        if 0 <= slide_index < len(self.containers):
            col1, col2, col3 = st.columns([0.6, 8, 0.6])  # Adjust column proportions for spacing

            # "Prev" button on the left
            with col1:
                st.button("❮", key="prev_button", on_click=self.go_to_previous_slide)

            # Heading in the center
            with col2:
                st.markdown(
                    f"<h2 style='text-align: center; margin: 0;'>{self.slide_titles[slide_index]}</h2>",
                    unsafe_allow_html=True,
                )

            # "Next" button on the right
            with col3:
                st.button("❯", key="next_button", on_click=self.go_to_next_slide)

            # Render the slide content
            self.containers[slide_index]()  # Call the container function
        else:
            st.error("Slide index out of range!")

    def show(self):
        """
        Show the slideshow. Displays the current slide and renders the controls.
        
        This method is the primary interface for displaying the slideshow within the
        Streamlit application. It invokes the `render_slide` method to display the
        appropriate slide content and navigation buttons.
        """
        self.render_slide()
