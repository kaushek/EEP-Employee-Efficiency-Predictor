from turtle import st
import streamlit as st
from predict_page import show_predict_page
from explore_page import ShowDashboardPage

page = st.sidebar.selectbox("Dashboard or Predict", ("Predict", "Dashboard"))

if page == "Predict":
    show_predict_page()
else:
    ShowDashboardPage()