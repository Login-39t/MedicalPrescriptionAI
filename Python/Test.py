import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import load_data, plot_tasks_line, plot_scores_box, plot_hours_heatmap

st.set_page_config(page_title="Employee Productivity Dashboard", layout="wide")

st.title("Employee Productivity Dashboard")

# Load data
data_url = "productivity.csv"
df = load_data("C:/Users/LXGIN/OneDrive/Desktop/OOPS/Python/productivity.csv")

# Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Tasks Completed Over Time")
    plot_tasks_line(df)

with col2:
    st.subheader("Performance Scores by Team")
    plot_scores_box(df)

st.subheader("Work Hours Heatmap by Day of Week")
plot_hours_heatmap(df)





