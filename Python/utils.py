# utils.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def load_data(gcs_path):
    df = pd.read_csv(gcs_path)
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.day_name()
    return df

def plot_tasks_line(df):
    for employee in df['employee_name'].unique():
        emp_df = df[df['employee_name'] == employee].set_index('date')
        st.line_chart(emp_df['tasks_completed'])

def plot_scores_box(df):
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x='team', y='performance_score', ax=ax)
    st.pyplot(fig)

def plot_hours_heatmap(df):
    pivot = df.groupby(['employee_name', 'day_of_week'])['work_hours'].mean().unstack().fillna(0)
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    available_days = [day for day in days if day in pivot.columns]
    pivot = pivot[available_days]

