import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt

def ShortenCategory(Category, CutOffVal):
    categoryMap = {}
    for i in range(len(Category)):
        if (Category.values[i] >= CutOffVal):
            categoryMap[Category.index[i]] = Category.index[i]
        else:
            categoryMap[Category.index[i]] = "Other"
    return categoryMap

def LoadData():
    # df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
    df = pd.read_csv("../EEP.Data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    df = df[df["Attrition"] == "No"]
    df = df[["Department", "PerformanceRating", "JobRole", "PercentSalaryHike", "TotalWorkingYears", "PerformanceScore"]]
    return df

df = LoadData()

def LoadCalculatedData():
       df_cal = pd.read_csv("../EEP.Data/Calculated_Employee_Dataset.csv")
       df_cal = df_cal[['MonthlyRate', 'YearsSinceLastPromotion', 'YearsAtCompany',
                        'TotalWorkingYears', 'NumCompaniesWorked', 'PercentSalaryHike',
                        'PerformanceScore', 'MonthlyRate_cal', 'YearsAtCompany_cal',
                        'TotalWorkingYears_cal', 'YearsSinceLastPromotion_cal',
                        'NumCompaniesWorked_cal', 'Prediction_PercentSalaryHike', 'Prediction_PercentSalaryHike_cal']]
       return df_cal

df_cal = LoadCalculatedData()

def ShowDashboardPage():
    st.title("EEP Dashboard")
    data = df_cal.groupby(["YearsSinceLastPromotion_cal"])["YearsSinceLastPromotion_cal"].count().sort_values(ascending=True)
    fig, ax = plt.subplots()
    ax.pie(data, labels=(data.index + ' (' + data.map(str)+ ')'), wedgeprops = { 'linewidth' : 10, 'edgecolor' : 'white'})
    p = plt.gcf()
    p.gca().add_artist(plt.Circle( (0,0), 0.7, color='white'))
    st.write("""##### Headcount on the years since last promotion""")
    st.pyplot(fig)

    data = df_cal.groupby(["MonthlyRate_cal"])["Prediction_PercentSalaryHike"].mean().sort_values(ascending=True)
    fig, ax = plt.subplots()
    ax.pie(data, labels=data.index, autopct="%1.1f%%", startangle=90)
    ax.axis("equal")
    st.write("""##### Monthly Rate wise Predicted Salary Hike""")
    st.pyplot(fig)

    # lesstentotworkyrs = df_cal[df_cal['TotalWorkingYears'] < 5] 
    # st.write(lesstentotworkyrs)
    # chart_data = pd.DataFrame(
    # lesstentotworkyrs["TotalWorkingYears"],
    # df_cal["Prediction_PercentSalaryHike"])
    # st.bar_chart(chart_data)

    # maxPerfHike = df_cal['Prediction_PercentSalaryHike'].max()
    # maxPerfHike = int(maxPerfHike)
    # print(maxPerfHike)


    
    # st.write(df_cal)

    MonthlyRate = df_cal['MonthlyRate'].values
    YearsSinceLastPromotion = df_cal['YearsSinceLastPromotion'].values
    YearsAtCompany = df_cal['YearsAtCompany'].values
    TotalWorkingYears = df_cal['TotalWorkingYears'].values
    NumCompaniesWorked = df_cal['NumCompaniesWorked'].values
    Prediction_PercentSalaryHike = df_cal['Prediction_PercentSalaryHike'].values

    chart_data = pd.DataFrame(
    #  np.random.randn(20, 3),
    #  columns=['a', 'b', 'c'])
        {
            'MonthlyRate':MonthlyRate,
            'YearsSinceLastPromotion':YearsSinceLastPromotion,
            'YearsAtCompany':YearsAtCompany,
            'TotalWorkingYears': TotalWorkingYears,
            'NumCompaniesWorked':NumCompaniesWorked,
            'Prediction_PercentSalaryHike':Prediction_PercentSalaryHike
        },
        columns=['MonthlyRate', 'YearsSinceLastPromotion', 'YearsAtCompany',
                        'TotalWorkingYears', 'NumCompaniesWorked', 'Prediction_PercentSalaryHike'])
                        
    # st.write(chart_data)       

    chart_data = chart_data.melt('Prediction_PercentSalaryHike', var_name='name', value_name='value')
    # st.write(chart_data)  
    # st.line_chart(chart_data)
    chart = alt.Chart(chart_data).mark_line().encode(
    x=alt.X('Prediction_PercentSalaryHike:N'),
    y=alt.Y('value:Q'),
    color=alt.Color("name:N")
    ).properties(title="Hello World")
    st.write("""##### """)
    st.write("""##### Impact on the variance of Inputs with the predicted salary hike""")
    st.altair_chart(chart, use_container_width=True)


    # crossTab = pd.crosstab(df_cal['Prediction_PercentSalaryHike'], df_cal['TotalWorkingYears_cal'])
    # print(crossTab)

    data = df.groupby(["Department"])["PerformanceRating"].mean().sort_values(ascending=True)
    fig, ax = plt.subplots()
    ax.pie(data, labels=data.index, autopct="%1.1f%%", startangle=90)
    ax.axis("equal")
    st.write("""##### Department wise average performance""")
    st.pyplot(fig)

    data = df.groupby(["JobRole"])["PercentSalaryHike"].median().sort_values(ascending=True)
    st.write("""##### Job role wise average percentage salary hike""")
    st.bar_chart(data)

    data = df.groupby(["Department"])["PercentSalaryHike"].mean().sort_values(ascending=True)
    st.write("""##### Department wise average percentage salary hike""")
    st.bar_chart(data)

    