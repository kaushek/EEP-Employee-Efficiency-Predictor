from turtle import st
import streamlit as st
import streamlit_authenticator as stauth
from predict_page import show_predict_page
from explore_page import ShowDashboardPage
import base64

names = ['HR Manager','HR Executives']
usernames = ['Manager','Executives']
passwords = ['123','456']

hashed_passwords = stauth.Hasher(passwords).generate()

authenticator = stauth.Authenticate(names,usernames,hashed_passwords,
    'login','login_key',cookie_expiry_days=60)

# st.title("Employee Efficiency Predictor")
st.markdown("<h1 style='text-align: center; color: white;'>Employee Efficiency Predictor</h1>", unsafe_allow_html=True)

name, authentication_status, username = authenticator.login('Login','main')

if authentication_status:
    authenticator.logout('Logout', 'main')
    st.write('Welcome *%s*' % (name))
    page = st.sidebar.selectbox("SELECTION", ("Predict", "Dashboard"))

    if page == "Predict":
        show_predict_page()
    else:
        ShowDashboardPage()
elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')

