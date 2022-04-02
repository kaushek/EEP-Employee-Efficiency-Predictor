import streamlit as st
import pickle 
import numpy as np
import requests
import json

def load_model():
    with open('../EEP.Data/EEP_MLScript_PerformanceScore.pkl', 'rb') as file:
        pscdata = pickle.load(file)
    return pscdata

def load_PrecentageSalHike_Model():   
    with open('../EEP.Data/EEP_MLScript_PercentSalaryHike.pkl', 'rb') as file:
        pshdata = pickle.load(file)
    return pshdata

pscdata = load_model()
pshdata = load_PrecentageSalHike_Model()

regressor_perfScore = pscdata["model"]
lblMonthlyRate_perfScore = pscdata["lblMonthlyRate"]
lblYearsAtCompany_perfScore = pscdata["lblYearsAtCompany"]
lblTotWorkYear_perfScore = pscdata["lblTotWorkYear"]
lblYearsSinceLastPromotion_perfScore = pscdata["lblYearsSinceLastPromotion"]
lblNumCompaniesWorked_perfScore = pscdata["lblNumCompaniesWorked"]

regressor_perHike = pshdata["model"]
lblMonthlyRate_perHike = pshdata["lblMonthlyRate"]
lblYearsAtCompany_perHike = pshdata["lblYearsAtCompany"]
lblTotWorkYear_perHike = pshdata["lblTotWorkYear"]
lblYearsSinceLastPromotion_perHike = pshdata["lblYearsSinceLastPromotion"]
lblNumCompaniesWorked_perHike = pshdata["lblNumCompaniesWorked"]
    
def show_predict_page():
    st.title("Employee Efficiency Predictor")

    st.write("""### Input Information. """)

    MonthlyRate = {
        'less than 2500', '2500 & above', '5000 & above', '7500 & above', '10000 & above', 
        '12500 & above', '15000 & above', '17500 & above', '20000 & above',  
        '22500 & above', '25000 & above' 
    }

    YearsAtCompany = {
        'less than 2 years', 'More than 2 years', 'More than 3 years', 'More than 5 years', 'More than 7 years', 
        'More than 10 years', 'More than 12 years', 'More than 15 years', 'More than 17 years', 'More than 20 years',
        'More than 22 years', 'More than 25 years', 'More than 27 years', 'More than 30 years', 'More than 32 years',
        'More than 35 years'
    }

    TotalWorkYears = {
        'less than 2 years', 'More than 2 years', 'More than 3 years', 'More than 5 years', 'More than 7 years', 
        'More than 10 years', 'More than 12 years', 'More than 15 years', 'More than 17 years', 'More than 20 years',
        'More than 22 years', 'More than 25 years', 'More than 27 years', 'More than 30 years', 'More than 32 years',
        'More than 35 years'      
    }

    YearsSinceLastPromotion = {
        'less than 1 year', 'less than 2 years', 'less than 3 years', 'less than 4 years', 'less than 5 years', 
        'less than 6 years', 'over 6 years' 
    }

    NumCompaniesWorked = {
        'less than 1 year', 'less than 2 years', 'less than 3 years', 'less than 4 years', 'less than 5 years', 
        'less than 6 years', 'less than 7 years', 'less than 8 years', 'less than 9 years', 'over 10 years'
    }

    monthrate = st.selectbox("Monthly Rate", MonthlyRate)
    yratcomp = st.selectbox("Years worked in the company", YearsAtCompany)
    totworkyr = st.selectbox("Total Work Years", TotalWorkYears)  
    yrslstpromo = st.selectbox("Years since last promotion", YearsSinceLastPromotion)
    numofcomp = st.selectbox("Total training times last year", NumCompaniesWorked)
   
    ok = st.button("Calculate Rate")
    if ok:
        x_perHike = np.array([[monthrate, yratcomp, totworkyr, yrslstpromo, numofcomp]])
        x_perHike[:, 0] = lblMonthlyRate_perHike.transform(x_perHike[:, 0])
        x_perHike[:, 1] = lblYearsAtCompany_perHike.transform(x_perHike[:, 1])
        x_perHike[:, 2] = lblTotWorkYear_perHike.transform(x_perHike[:, 2])
        x_perHike[:, 3] = lblYearsSinceLastPromotion_perHike.transform(x_perHike[:, 3])                                            
        x_perHike[:, 4] = lblNumCompaniesWorked_perHike.transform(x_perHike[:, 4])
        x_perHike =  x_perHike.astype(float)

        x_perScore = np.array([[monthrate, yratcomp, totworkyr, yrslstpromo, numofcomp]])
        x_perScore[:, 0] = lblMonthlyRate_perfScore.transform(x_perScore[:, 0])
        x_perScore[:, 1] = lblYearsAtCompany_perfScore.transform(x_perScore[:, 1])
        x_perScore[:, 2] = lblTotWorkYear_perfScore.transform(x_perScore[:, 2])
        x_perScore[:, 3] = lblYearsSinceLastPromotion_perfScore.transform(x_perScore[:, 3])                                            
        x_perScore[:, 4] = lblNumCompaniesWorked_perfScore.transform(x_perScore[:, 4])
        x_perScore =  x_perScore.astype(float)

        percentage =  regressor_perHike.predict(x_perHike)
        score = regressor_perfScore.predict(x_perScore)

        st.subheader(f"Predicted increment percentage: {percentage[0]:.2f}" + "%")
        st.subheader(f"Predicted performance score: {score[0]:.2f}")



