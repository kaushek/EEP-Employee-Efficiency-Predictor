import streamlit as st
import pickle 
import numpy as np
from memory_profiler import profile

# Loading Performance Score Model from the local storage
def load_PerformanceScore_Model():
    with open('../EEP.Data/EEP_MLScript_PerformanceScore.pkl', 'rb') as file:
        pscdata = pickle.load(file)
    return pscdata

# Loading Percentage Salary Hike from the local storage
def load_PrecentageSalHike_Model():   
    with open('../EEP.Data/EEP_MLScript_PercentSalaryHike.pkl', 'rb') as file:
        pshdata = pickle.load(file)
    return pshdata

pscdata = load_PerformanceScore_Model()
pshdata = load_PrecentageSalHike_Model()

# Assigning the performance score regressor model and the labels to variables
regressor_perfScore = pscdata["model"]
lblMonthlyIncome_perfScore = pscdata["lblMonthlyIncome"]
lblMonthlyRate_perfScore = pscdata["lblMonthlyRate"]
lblDailyRate_perfScore = pscdata["lblDailyRate"]
lblYearsAtCompany_perfScore = pscdata["lblYearsAtCompany"]
lblTotWorkYear_perfScore = pscdata["lblTotWorkYear"]
lblYearsSinceLastPromotion_perfScore = pscdata["lblYearsSinceLastPromotion"]
lblYearsWithCurrManager_perfScore = pscdata["lblYearsWithCurrManager"]
lblNumCompaniesWorked_perfScore = pscdata["lblNumCompaniesWorked"]

# Assigning the percentage salary hike regressor model and the labels to variables
regressor_perHike = pshdata["model"]
lblMonthlyIncome_perHike = pshdata["lblMonthlyIncome"]
lblMonthlyRate_perHike = pshdata["lblMonthlyRate"]
lblDailyRate_perHike = pshdata["lblDailyRate"]
lblYearsAtCompany_perHike = pshdata["lblYearsAtCompany"]
lblTotWorkYear_perHike = pshdata["lblTotWorkYear"]
lblYearsSinceLastPromotion_perHike = pshdata["lblYearsSinceLastPromotion"]
lblYearsWithCurrManager_perHike = pshdata["lblYearsWithCurrManager"]
lblNumCompaniesWorked_perHike = pshdata["lblNumCompaniesWorked"]

MonthlyInc = ("less than 2500", "2500 & above", "5000 & above")

MonthRate = ("less than 5000", "5000 & above", "10000 & above", "15000 & above", "20000 & above", "25000 & above")

DailyRate = ("less than 250", "250 & above", "500 & above", "750 & above", "1000 & above", "1250 & above" )

YrAtComp = { 1: "less than 3 years", 2: "More than 3 years", 3: "More than 6 years", 4: "More than 9 years" }
def format_func_yac(option):
    return YrAtComp[option]

TotWorkYr = { 1: "less than 3 years", 2: "More than 3 years",  3: "More than 6 years", 4: "More than 9 years", 5: "More than 12 years", 6: "More than 15 years" }
def format_func_twy(option):
    return TotWorkYr[option]

YrsLastPromo = { 1: "0 years", 2: "1 year", 3: "2 years" }
def format_func_ylp(option):
    return YrsLastPromo[option]

YrsWithCurrMan = {1: "less than 3 years", 2: "More than 3 years", 3: "More than 6 years"}
def format_func_ywcm(option):
    return YrsWithCurrMan[option]

NumOfComp = { 1: "less than 2", 2: "less than 4", 3: "less than 6", 4: "less than 8"}
def format_func_noc(option):
    return NumOfComp[option]

@profile    
def show_predict_page():
    st.write("""### Input Information: """)

    # Assigning selected values from the drop down
    monthinc = st.selectbox("Monthly Income", MonthlyInc)
    monthrate = st.selectbox("Monthly Rate", MonthRate)
    dailyate = st.selectbox("Daily Rate", DailyRate)
    yratcomp = st.selectbox("Years at Company",  options=list(YrAtComp.keys()), format_func=format_func_yac)
    totworkyr = st.selectbox("Total Work Years", options=list(TotWorkYr.keys()), format_func=format_func_twy)  
    yrslstpromo = st.selectbox("Years Since Last Promotion", options=list(YrsLastPromo.keys()), format_func=format_func_ylp)
    yrswithcurrman = st.selectbox("Years with Current Manager", options=list(YrsWithCurrMan.keys()), format_func=format_func_ywcm)
    numofcomp = st.selectbox("Number of Companies Worked", options=list(NumOfComp.keys()), format_func=format_func_noc)
   
    if (totworkyr < yratcomp):
        st.error('Total working years cannot be lesser than years worked in company')
    if (yrswithcurrman > yratcomp):
        st.error('Years with current manager cannot be greater than years worked in company')
    # elif (yrslstpromo > yratcomp):
    #     st.error('Years since last promotion cannot be greater than years worked in company')
    # elif (yrslstpromo > totworkyr):
    #     st.error('Years since last promotion cannot be greater than total work years')
    elif (totworkyr < yrswithcurrman):
        st.error('Total working years cannot be lesser than years with current manager')
    elif (yrswithcurrman > totworkyr):
        st.error('Years with current manager cannot be greater than total work years')
        
    else:

    # Passing the selected values into a numpy array
    # Transformiing the lables and performing predictions
        ok = st.button("Calculate Rate")
        if ok:
            x_perHike = np.array([[monthinc, monthrate, dailyate, 
                format_func_yac(yratcomp), format_func_twy(totworkyr), format_func_ylp(yrslstpromo), format_func_ywcm(yrswithcurrman), format_func_noc(numofcomp)]])
            x_perHike[:, 0] = lblMonthlyIncome_perHike.transform(x_perHike[:, 0])
            x_perHike[:, 1] = lblMonthlyRate_perHike.transform(x_perHike[:, 1])
            x_perHike[:, 2] = lblDailyRate_perHike.transform(x_perHike[:, 2])
            x_perHike[:, 3] = lblYearsAtCompany_perHike.transform(x_perHike[:, 3])
            x_perHike[:, 4] = lblTotWorkYear_perHike.transform(x_perHike[:, 4])
            x_perHike[:, 5] = lblYearsSinceLastPromotion_perHike.transform(x_perHike[:, 5])                                            
            x_perHike[:, 6] = lblYearsWithCurrManager_perHike.transform(x_perHike[:, 6])                                            
            x_perHike[:, 7] = lblNumCompaniesWorked_perHike.transform(x_perHike[:, 7])
            x_perHike =  x_perHike.astype(float)

            x_perScore = np.array([[monthinc, monthrate, dailyate, 
                format_func_yac(yratcomp), format_func_twy(totworkyr), format_func_ylp(yrslstpromo), format_func_ywcm(yrswithcurrman), format_func_noc(numofcomp)]])
            x_perScore[:, 0] = lblMonthlyIncome_perfScore.transform(x_perScore[:, 0])
            x_perScore[:, 1] = lblMonthlyRate_perfScore.transform(x_perScore[:, 1])
            x_perScore[:, 2] = lblDailyRate_perfScore.transform(x_perScore[:, 2])
            x_perScore[:, 3] = lblYearsAtCompany_perfScore.transform(x_perScore[:, 3])
            x_perScore[:, 4] = lblTotWorkYear_perfScore.transform(x_perScore[:, 4])
            x_perScore[:, 5] = lblYearsSinceLastPromotion_perfScore.transform(x_perScore[:, 5])                                            
            x_perScore[:, 6] = lblYearsWithCurrManager_perfScore.transform(x_perScore[:, 6])                                            
            x_perScore[:, 7] = lblNumCompaniesWorked_perfScore.transform(x_perScore[:, 7])
            x_perScore =  x_perScore.astype(float)

            percentage =  regressor_perHike.predict(x_perHike)
            score = regressor_perfScore.predict(x_perScore)

            st.subheader(f"Predicted increment percentage: {percentage[0]:.2f}" + "%")
            st.subheader(f"Predicted performance score: {score[0]:.2f}")

show_predict_page()