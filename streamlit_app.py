import streamlit as st
import hw1, hw2

st.set_page_config(page_title="HW-Manager", layout="wide")

nav = st.navigation({
    "Labs": [
        st.Page(lab1.run, title="HW 1", url_path="hw1"),
        st.Page(lab2.run, title="HW 2 (Default)", url_path="hw2", default=True),
    ]
})

nav.run()