import streamlit as st
import pickle 
import numpy as np

#Loading Performance Score Model from the local storage
def load_PerformanceScore_Model():
    with open('../EEP.Data/EEP_MLScript_PerformanceScore.pkl', 'rb') as file:
        pscdata = pickle.load(file)
    return pscdata

#Loading Percentage Salary Hike from the local storage
def load_PrecentageSalHike_Model():   
    with open('../EEP.Data/EEP_MLScript_PercentSalaryHike.pkl', 'rb') as file:
        pshdata = pickle.load(file)
    return pshdata

pscdata = load_PerformanceScore_Model()
pshdata = load_PrecentageSalHike_Model()

#Assigning the performance score regressor model and the labels to variables
regressor_perfScore = pscdata["model"]
lblMonthlyIncome_perfScore = pscdata["lblMonthlyIncome"]
lblMonthlyRate_perfScore = pscdata["lblMonthlyRate"]
lblDailyRate_perfScore = pscdata["lblDailyRate"]
lblYearsAtCompany_perfScore = pscdata["lblYearsAtCompany"]
lblTotWorkYear_perfScore = pscdata["lblTotWorkYear"]
lblYearsSinceLastPromotion_perfScore = pscdata["lblYearsSinceLastPromotion"]
lblNumCompaniesWorked_perfScore = pscdata["lblNumCompaniesWorked"]

#Assigning the percentage salary hike regressor model and the labels to variables
regressor_perHike = pshdata["model"]
lblMonthlyIncome_perHike = pshdata["lblMonthlyIncome"]
lblMonthlyRate_perHike = pshdata["lblMonthlyRate"]
lblDailyRate_perHike = pshdata["lblDailyRate"]
lblYearsAtCompany_perHike = pshdata["lblYearsAtCompany"]
lblTotWorkYear_perHike = pshdata["lblTotWorkYear"]
lblYearsSinceLastPromotion_perHike = pshdata["lblYearsSinceLastPromotion"]
lblNumCompaniesWorked_perHike = pshdata["lblNumCompaniesWorked"]

MonthlyInc = ("less than 1500", "1500 & above", "2500 & above", "3500 & above", "4500 & above", "5500 & above", "6500 & above", 
              "7500 & above", "8500 & above", "9500 & above", "10500 & above", "11500 & above", "12500 & above")

MonthRate = ("less than 2500", "2500 & above", "5000 & above", "7500 & above", "10000 & above", "12500 & above", "15000 & above", 
             "17500 & above", "20000 & above", "22500 & above", "25000 & above")

DailyRate = ("less than 200", "200 & above", "300 & above", "400 & above", "500 & above", "600 & above", "700 & above", 
             "800 & above", "900 & above", "1000 & above", "1100 & above", "1200 & above", "1300 & above", "1400 & above")

YearsAtComp = { 1: "less than 2 years", 2: "More than 2 years", 3: "More than 3 years", 4: "More than 5 years", 5: "More than 7 years", 
                6: "More than 10 years", 7: "More than 12 years", 8: "More than 15 years", 9: "More than 17 years"}
def format_func_yac(option):
    return YearsAtComp[option]

TotWorkYr = {   1: "less than 2 years", 2: "More than 2 years", 3: "More than 3 years", 4: "More than 5 years", 5: "More than 7 years", 
                6: "More than 10 years", 7: "More than 12 years", 8: "More than 15 years", 9: "More than 17 years" }
def format_func_twy(option):
    return TotWorkYr[option]

YrsLastPromo = {1: "less than 1 year", 2: "less than 2 years", 3: "less than 3 years", 4: "less than 4 years", 5: "less than 5 years", 
                6: "less than 6 years", 7: "over 6 years" }
def format_func_ylp(option):
    return YrsLastPromo[option]

# NumOfComp = {1: "less than 1", 2: "less than 2", 3: "less than 3 years", 4: "less than 4 years", 5: "less than 5 years", 
#              6: "less than 6 years", 7: "less than 7 years", 8: "less than 8 years", 9: "less than 9 years", 10: "over 10 years"}
NumOfComp = {1: "less than 2", 2: "less than 4", 3: "less than 6", 4: "less than 8", 5: "less than 10", 6: "over 10"}
def format_func_noc(option):
    return NumOfComp[option]
    
def show_predict_page():
    st.write("""### Input Information: """)

    #Assigning selected values from the drop down
    monthinc = st.selectbox("Monthly Income", MonthlyInc)
    monthrate = st.selectbox("Monthly Rate", MonthRate)
    dailyate = st.selectbox("Daily Rate", DailyRate)
    yratcomp = st.selectbox("Years worked in the company", options=list(YearsAtComp.keys()), format_func=format_func_yac)
    totworkyr = st.selectbox("Total Work Years", options=list(TotWorkYr.keys()), format_func=format_func_twy)  
    yrslstpromo = st.selectbox("Years since last promotion", options=list(YrsLastPromo.keys()), format_func=format_func_ylp)
    numofcomp = st.selectbox("Number of Companies Worked", options=list(NumOfComp.keys()), format_func=format_func_noc)
   
    if (totworkyr < yratcomp):
        st.error('Total working years cannot be lesser than years worked in company')
    elif (yrslstpromo > yratcomp):
        st.error('Years since last promotion cannot be greater than years worked in company')
    elif (yrslstpromo > totworkyr):
        st.error('Years since last promotion cannot be greater than total work years')
    else:

    #Passing the selected values into a numpy array
    #Transformiing the lables and performing predictions
        ok = st.button("Calculate Rate")
        if ok:
            x_perHike = np.array([[monthinc, monthrate, dailyate, format_func_yac(yratcomp), format_func_twy(totworkyr), format_func_ylp(yrslstpromo), format_func_noc(numofcomp)]])
            x_perHike[:, 0] = lblMonthlyIncome_perHike.transform(x_perHike[:, 0])
            x_perHike[:, 1] = lblMonthlyRate_perHike.transform(x_perHike[:, 1])
            x_perHike[:, 2] = lblDailyRate_perHike.transform(x_perHike[:, 2])
            x_perHike[:, 3] = lblYearsAtCompany_perHike.transform(x_perHike[:, 3])
            x_perHike[:, 4] = lblTotWorkYear_perHike.transform(x_perHike[:, 4])
            x_perHike[:, 5] = lblYearsSinceLastPromotion_perHike.transform(x_perHike[:, 5])                                            
            x_perHike[:, 6] = lblNumCompaniesWorked_perHike.transform(x_perHike[:, 6])
            x_perHike =  x_perHike.astype(float)

            x_perScore = np.array([[monthinc, monthrate, dailyate, format_func_yac(yratcomp), format_func_twy(totworkyr), format_func_ylp(yrslstpromo), format_func_noc(numofcomp)]])
            x_perScore[:, 0] = lblMonthlyIncome_perfScore.transform(x_perScore[:, 0])
            x_perScore[:, 1] = lblMonthlyRate_perfScore.transform(x_perScore[:, 1])
            x_perScore[:, 2] = lblDailyRate_perfScore.transform(x_perScore[:, 2])
            x_perScore[:, 3] = lblYearsAtCompany_perfScore.transform(x_perScore[:, 3])
            x_perScore[:, 4] = lblTotWorkYear_perfScore.transform(x_perScore[:, 4])
            x_perScore[:, 5] = lblYearsSinceLastPromotion_perfScore.transform(x_perScore[:, 5])                                            
            x_perScore[:, 6] = lblNumCompaniesWorked_perfScore.transform(x_perScore[:, 6])
            x_perScore =  x_perScore.astype(float)

            percentage =  regressor_perHike.predict(x_perHike)
            score = regressor_perfScore.predict(x_perScore)

            st.subheader(f"Predicted increment percentage: {percentage[0]:.2f}" + "%")
            st.subheader(f"Predicted performance score: {score[0]:.2f}")
