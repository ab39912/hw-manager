import streamlit as st
import lab1, lab2, lab3, lab4, lab5

st.set_page_config(page_title="Document QA", layout="wide")

nav = st.navigation({
    "HW": [
        st.Page(lab1.run, title="HW 1", url_path="lab-1"),
        st.Page(lab2.run, title="HW 2 ", url_path="lab-2"),
        st.Page(lab3.run, title="HW 3 ", url_path="lab-3"),
        st.Page(lab4.run, title="HW 4 ", url_path="lab-4"),
        st.Page(lab5.run, title="HW 5 ", url_path="lab-5"),
        st.Page(lab7.run, title="HW 7 ", url_path="lab-7", default=True)
    ]
})

nav.run()