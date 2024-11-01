import streamlit as st

def set_page_config(page_title, page_icon, layout="wide", initial_sidebar_state="expanded", menu_items=None):
    if menu_items is None:
        menu_items = {
            'Get Help': 'https://www.linkedin.com/in/kirchhoff-kevin/'
        }
    
    st.set_page_config(
        page_title=page_title,
        page_icon=page_icon,
        layout=layout,
        initial_sidebar_state=initial_sidebar_state,
        menu_items=menu_items
    )

    st.logo("https://www.claneo.com/wp-content/uploads/Element-4.svg")