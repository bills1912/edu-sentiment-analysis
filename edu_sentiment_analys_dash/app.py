import streamlit as st
import streamlit as st
from multiapp import MultiApp
from apps import (
    home,
)
st.set_page_config(
        page_title="Analisis Sentimen tentang Pendidikan di Indonesia",
        page_icon=":speech_balloon:",
        layout="wide"
    )

apps = MultiApp()

# Add all your application here

apps.add_app("Home", home.app)

# The main app
apps.run()