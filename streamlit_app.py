import streamlit as st
import lab1, lab2, lab3

st.set_page_config(page_title="Document QA", layout="wide")

nav = st.navigation({
    "HW": [
        st.Page(lab1.run, title="HW 1", url_path="lab-1"),
        st.Page(lab2.run, title="HW 2 ", url_path="lab-2"),
        st.Page(lab3.run, title="HW 3 (Default)", url_path="lab-3", default=True)
    ]
})

nav.run()