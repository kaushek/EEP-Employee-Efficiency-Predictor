from unicodedata import name
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

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
       df_cal = df_cal[['MonthlyIncome', 'MonthlyRate', 'DailyRate', 'YearsSinceLastPromotion',
                        'YearsAtCompany', 'TotalWorkingYears', 'NumCompaniesWorked',
                        'PercentSalaryHike', 'PerformanceScore', 'MonthlyIncome_cal',
                        'MonthlyRate_cal', 'DailyRate_cal', 'YearsAtCompany_cal',
                        'TotalWorkingYears_cal', 'YearsSinceLastPromotion_cal',
                        'NumCompaniesWorked_cal', 'Prediction_PercentSalaryHike',
                        'Prediction_PerformanceScore', 'Prediction_PercentSalaryHike_cal', 'Prediction_PerformanceScore_cal']]
       df_cal.reset_index(inplace=True)
       return df_cal

df_cal = LoadCalculatedData()

def ShowDashboardPage():
    st.title("EEP Dashboard")
    data = df_cal.groupby(["Prediction_PercentSalaryHike_cal"])["Prediction_PercentSalaryHike_cal"].count().sort_values(ascending=True)
    fig, ax = plt.subplots()
    ax.pie(data, labels=(data.index + ' (' + data.map(str)+ ')'), wedgeprops = { 'linewidth' : 10, 'edgecolor' : 'white'})
    p = plt.gcf()
    p.gca().add_artist(plt.Circle( (0,0), 0.7, color='white'))
    st.write("""##### Headcount on the percentage salary hike""")
    st.pyplot(fig)

    data = df_cal.groupby(["Prediction_PerformanceScore_cal"])["Prediction_PerformanceScore_cal"].count().sort_values(ascending=True)
    fig, ax = plt.subplots()
    ax.pie(data, labels=(data.index + ' (' + data.map(str)+ ')'), wedgeprops = { 'linewidth' : 10, 'edgecolor' : 'white'})
    p = plt.gcf()
    p.gca().add_artist(plt.Circle( (0,0), 0.7, color='white'))
    st.write("""##### Headcount on the performance score""")
    st.pyplot(fig)

    data = df_cal.groupby(["YearsSinceLastPromotion_cal"])["YearsSinceLastPromotion_cal"].count().sort_values(ascending=True)
    fig, ax = plt.subplots()
    ax.pie(data, labels=(data.index + ' (' + data.map(str)+ ')'), wedgeprops = { 'linewidth' : 10, 'edgecolor' : 'white'})
    p = plt.gcf()
    p.gca().add_artist(plt.Circle( (0,0), 0.7, color='white'))
    st.write("""##### Headcount on the years since last promotion""")
    st.pyplot(fig)

    data = df_cal.groupby(["MonthlyRate_cal"])["Prediction_PerformanceScore_cal"].count().sort_values(ascending=True)
    st.write("""##### Years since last promotion count""")
    st.bar_chart(data)

    data = df_cal.groupby(["MonthlyIncome_cal"])["Prediction_PercentSalaryHike"].mean().sort_values(ascending=True)
    fig, ax = plt.subplots()
    ax.pie(data, labels=data.index, autopct="%1.1f%%", startangle=90)
    ax.axis("equal")
    st.write("""##### Monthly Rate wise Predicted Salary Hike""")
    st.pyplot(fig)

    data = df_cal.groupby(["MonthlyIncome_cal"])["Prediction_PerformanceScore"].mean().sort_values(ascending=True)
    fig, ax = plt.subplots()
    ax.pie(data, labels=data.index, autopct="%1.1f%%", startangle=90)
    ax.axis("equal")
    st.write("""##### Monthly rate wise predicted performance score""")
    st.pyplot(fig)

    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=df_cal['MonthlyIncome'], y=df_cal['Prediction_PercentSalaryHike'], name='MonthInc'))
    # fig.add_trace(go.Scatter(x=df_cal['MonthlyIncome'], y=df_cal['Prediction_PercentSalaryHike'], name='SalaryHike'))
    # fig.layout.update(title_text="Prediction ranges with total working years", xaxis_rangeslider_visible=True)
    # st.plotly_chart(fig)

  
    # MonthlyRate = df_cal['MonthlyRate'].values
    # YearsSinceLastPromotion = df_cal['YearsSinceLastPromotion'].values
    # YearsAtCompany = df_cal['YearsAtCompany'].values
    # TotalWorkingYears = df_cal['TotalWorkingYears'].values
    # NumCompaniesWorked = df_cal['NumCompaniesWorked'].values
    # Prediction_PercentSalaryHike = df_cal['Prediction_PercentSalaryHike'].values

    # chart_data = pd.DataFrame(
    # {
    #     'MonthlyRate':MonthlyRate,
    #     'YearsSinceLastPromotion':YearsSinceLastPromotion,
    #     'YearsAtCompany':YearsAtCompany,
    #     'TotalWorkingYears': TotalWorkingYears,
    #     'NumCompaniesWorked':NumCompaniesWorked,
    #     'Prediction_PercentSalaryHike':Prediction_PercentSalaryHike
    # },
    # columns=['MonthlyRate', 'YearsSinceLastPromotion', 'YearsAtCompany', 'TotalWorkingYears', 'NumCompaniesWorked', 'Prediction_PercentSalaryHike'])

    # chart_data = chart_data.melt('Prediction_PercentSalaryHike', var_name='name', value_name='value')
    # st.write(chart_data)
    # chart = alt.Chart(chart_data).mark_line().encode(
    # x=alt.X('Prediction_PercentSalaryHike'),
    # y=alt.Y('value'),
    # color=alt.Color("name")
    # ).properties(title="Hello World")
    # st.write("""##### """)
    # st.write("""##### Impact on the variance of Inputs with the predicted salary hike""")
    # st.altair_chart(chart, use_container_width=True)


    # data = df.groupby(["Department"])["PerformanceRating"].mean().sort_values(ascending=True)
    # fig, ax = plt.subplots()
    # ax.pie(data, labels=data.index, autopct="%1.1f%%", startangle=90)
    # ax.axis("equal")
    # st.write("""##### Department wise average performance""")
    # st.pyplot(fig)

    # data = df.groupby(["JobRole"])["PercentSalaryHike"].median().sort_values(ascending=True)
    # st.write("""##### Job role wise average percentage salary hike""")
    # st.bar_chart(data)

    # data = df.groupby(["Department"])["PercentSalaryHike"].mean().sort_values(ascending=True)
    # st.write("""##### Department wise average percentage salary hike""")
    # st.bar_chart(data)

    # fig = plt.figure(figsize=(10, 4))
    # sns.lineplot(x='MonthlyIncome', y='Prediction_PercentSalaryHike', data=df_cal)
    # st.pyplot(fig)

    # fig = plt.figure(figsize=(10, 4))
    # plt.plot(df_cal['MonthlyIncome'], df_cal['Prediction_PercentSalaryHike'])
    # plt.xlabel('MonthlyIncome')
    # plt.ylabel('Prediction_PercentSalaryHike')
    # st.pyplot(fig)

    