from turtle import st
import streamlit as st
import streamlit_authenticator as stauth
from predict_page import show_predict_page
from explore_page import ShowDashboardPage
import base64

# Defining usernames and passwords of the system
names = ['HR Manager','HR Executives']
usernames = ['Manager','Executives']
passwords = ['123','456']

hashed_passwords = stauth.Hasher(passwords).generate()

authenticator = stauth.Authenticate(names,usernames,hashed_passwords,
    'login','login_key',cookie_expiry_days=60)

st.markdown("<h1 style='text-align: center; color: white;'>Employee Efficiency Predictor</h1>", unsafe_allow_html=True)

# Calling login method and getting authentication_status
name, authentication_status, username = authenticator.login('Login','main')

# Loading pages if the authentication_status is true
# Else error messages are shown
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

