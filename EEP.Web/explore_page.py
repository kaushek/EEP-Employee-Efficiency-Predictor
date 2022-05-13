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

# Loading the Dataset along with the predicted values
def LoadCalculatedData():
       df_cal = pd.read_csv("../EEP.Data/Calculated_Employee_Dataset.csv")
       df_cal = df_cal[['MonthlyRate', 'MonthlyIncome', 'DailyRate', 'NumCompaniesWorked',
                        'TotalWorkingYears', 'YearsWithCurrManager', 'YearsSinceLastPromotion',
                        'YearsAtCompany', 'PercentSalaryHike', 'PerformanceScore',
                        'MonthlyIncome_cal', 'MonthlyRate_cal', 'DailyRate_cal',
                        'YearsAtCompany_cal', 'TotalWorkingYears_cal',
                        'YearsSinceLastPromotion_cal', 'YearsWithCurrManager_cal',
                        'NumCompaniesWorked_cal', 'Prediction_PercentSalaryHike',
                        'Prediction_PercentSalaryHike_cal', 'Prediction_PerformanceScore',
                        'Prediction_PerformanceScore_cal']]
       df_cal.reset_index(inplace=True)
       return df_cal

df_cal = LoadCalculatedData()

def ShowDashboardPage():
    st.title("EEP Dashboard")

    # pie chart containing the predicted PercentSalaryHike count in each selected ranges
    data = df_cal.groupby(["Prediction_PercentSalaryHike_cal"])["Prediction_PercentSalaryHike_cal"].count().sort_values(ascending=True)
    fig, ax = plt.subplots()
    ax.pie(data, labels=(data.index + ' (' + data.map(str)+ ')'), wedgeprops = { 'linewidth' : 10, 'edgecolor' : 'white'})
    p = plt.gcf()
    p.gca().add_artist(plt.Circle( (0,0), 0.7, color='white'))
    st.write("""##### Headcount on the percentage salary hike""")
    st.pyplot(fig)

    # pie chart on the predicted PerformanceScore count in each selected ranges
    data = df_cal.groupby(["Prediction_PerformanceScore_cal"])["Prediction_PerformanceScore_cal"].count().sort_values(ascending=True)
    fig, ax = plt.subplots()
    ax.pie(data, labels=(data.index + ' (' + data.map(str)+ ')'), wedgeprops = { 'linewidth' : 10, 'edgecolor' : 'white'})
    p = plt.gcf()
    p.gca().add_artist(plt.Circle( (0,0), 0.7, color='white'))
    st.write("""##### Headcount on the performance score""")
    st.pyplot(fig)

    # pie chart showing the count of YearsSinceLastPromotion 
    data = df_cal.groupby(["YearsSinceLastPromotion_cal"])["YearsSinceLastPromotion_cal"].count().sort_values(ascending=True)
    fig, ax = plt.subplots()
    ax.pie(data, labels=(data.index + ' (' + data.map(str)+ ')'), wedgeprops = { 'linewidth' : 10, 'edgecolor' : 'white'})
    p = plt.gcf()
    p.gca().add_artist(plt.Circle( (0,0), 0.7, color='white'))
    st.write("""##### Headcount on the years since last promotion""")
    st.pyplot(fig)

    # bar chart showing predicted performance count among each monthly rate group
    data = df_cal.groupby(["MonthlyRate_cal"])["Prediction_PerformanceScore_cal"].count().sort_values(ascending=True)
    st.write("""##### Count on predicted performance score against monthly rate""")
    st.bar_chart(data)

    # bar chart showing predicted performance count among each monthly rate group
    data = df_cal.groupby(["TotalWorkingYears_cal"])["Prediction_PercentSalaryHike_cal"].count().sort_values(ascending=True)
    st.write("""##### Count on predicted salary hike against total working years""")
    st.bar_chart(data)

    # Monthly Rate wise Predicted Salary Hike
    data = df_cal.groupby(["MonthlyIncome_cal"])["Prediction_PercentSalaryHike"].mean().sort_values(ascending=True)
    fig, ax = plt.subplots()
    ax.pie(data, labels=data.index, autopct="%1.1f%%", startangle=90)
    ax.axis("equal")
    st.write("""##### Monthly Rate wise Predicted Salary Hike""")
    st.pyplot(fig)

    # Monthly rate wise predicted performance score
    data = df_cal.groupby(["MonthlyIncome_cal"])["Prediction_PerformanceScore"].mean().sort_values(ascending=True)
    fig, ax = plt.subplots()
    ax.pie(data, labels=data.index, autopct="%1.1f%%", startangle=90)
    ax.axis("equal")
    st.write("""##### Monthly rate wise predicted performance score""")
    st.pyplot(fig)

    