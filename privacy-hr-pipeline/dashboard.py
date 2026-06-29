import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Privacy-Preserving HR Analytics",
    page_icon="🔒",
    layout="wide"
)

st.title("🔒 Privacy-Preserving HR Analytics Pipeline")
st.markdown("A demonstration of k-anonymity, l-diversity, and differential privacy applied to HR data.")

@st.cache_data
def load_data():
    return pd.read_csv('hr_dataset.csv')

df = load_data()

col1, col2, col3 = st.columns(3)
col1.metric("Total Records", f"{len(df):,}")
col2.metric("Departments", df['department'].nunique())
col3.metric("Unique Zip Codes", f"{df['zip_code'].nunique():,}")

st.subheader("Sample Data")
st.dataframe(df.head(10))

st.subheader("Average Salary by Department")
st.bar_chart(df.groupby('department')['salary'].mean().sort_values())