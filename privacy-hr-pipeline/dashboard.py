# dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Privacy-Preserving HR Analytics",
    page_icon="🔒",
    layout="wide"
)

st.title("🔒 Privacy-Preserving HR Analytics Pipeline")
st.markdown("A demonstration of k-anonymity, l-diversity, and differential privacy applied to HR data.")

# load data (with caching for better performance)
@st.cache_data
def load_data():
    return pd.read_csv('hr_dataset.csv')

df = load_data()

# sidebar
st.sidebar.header("Privacy Parameters")
k_value = st.sidebar.slider("k-anonymity (k)", min_value=2, max_value=20, value=5)
l_value = st.sidebar.slider("l-diversity (l)", min_value=2, max_value=5, value=2)
epsilon_value = st.sidebar.slider(
    "Differential Privacy (ε)", 
    min_value=0.1, 
    max_value=10.0, 
    value=1.0, 
    step=0.1
)  

 # tabs
tab1, tab2, tab3, tab4 = st.tabs([ 
    "Original Data",
    "K-Anonymity", 
    "L-Diversity",
    "Differential Privacy"
])

with tab1:
    st.header("Original HR Dataset Overwiew")
    st.write("This is the original HR dataset before applying any privacy-preserving techniques.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Departments", df['department'].nunique())
    col3.metric("Unique Zip Codes", f"{df['zip_code'].nunique():,}")
    
st.subheader("Salary Distribution by Department")
dept_salary = df.groupby('department')['salary'].mean().sort_values()
st.bar_chart(dept_salary)
st.dataframe(df.head(10))

st.subheader("Sample Records")
st.dataframe(df.sample(10))

# helper functions from k-anonymity.py and l-diversity.py
def generalize_age(age, range_size=10):
    lower = (age // range_size) * range_size
    return f"{lower}-{lower + range_size - 1}"

def generalize_salary(salary, bucket_size=20000):
    lower = (salary // bucket_size) * bucket_size
    return f"${lower:,}-${lower + bucket_size - 1:,}"

def anonymize_no_zip(df, age_range=10):
    df = df.copy()
    df['age'] = df['age'].apply(lambda x: generalize_age(x, age_range))
    return df

def suppress_violations(df, qi_cols, k):
    group_sizes = df.groupby(qi_cols)[qi_cols[0]].transform('count')
    return df[group_sizes >= k].copy()

def check_k_anonymity(df, qi_cols, k):
    groups = df.groupby(qi_cols).size().reset_index(name='count')
    violations = groups[groups['count'] < k]
    return len(violations) == 0, len(violations), groups['count'].min()

def check_l_diversity(df, qi_cols, sensitive_col, l):
    groups = df.groupby(qi_cols)[sensitive_col].apply(
        lambda x: x.apply(generalize_salary).nunique()
    ).reset_index(name='distinct_values')
    violations = groups[groups['distinct_values'] < l]
    return len(violations) == 0, len(violations)

def laplace_mechanism(true_value, sensitivity, epsilon):
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale)
    return true_value + noise

def compute_sensitivity(df, column):
    return (df[column].max() - df[column].min()) / len(df)

# Tab2 k-anonymity
with tab2:
    st.header("K-Anonymity")
    st.markdown("""
    **What it does:** Ensures every record is indistinguishable from 
    at least k-1 others based on quasi-identifiers.
    
    **Quasi-identifiers used:** Age (generalized to ranges), 
    Gender, Department. Zip code excluded — nearly unique (4,873/5,000).
    """)

    QI = ['age', 'gender', 'department']
    anon_df = anonymize_no_zip(df, age_range=10)
    anon_suppressed = suppress_violations(anon_df, QI, k_value)

    satisfies, violations, min_group = check_k_anonymity(
        anon_suppressed, QI, k_value
    )
    suppressed_count = len(df) - len(anon_suppressed)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("k value", k_value)
    col2.metric(
        "K-Anonymity Satisfied",
        "✅ Yes" if satisfies else "❌ No"
    )
    col3.metric("Records Retained",
        f"{len(anon_suppressed):,} ({len(anon_suppressed)/len(df):.1%})"
    )
    col4.metric("Records Suppressed", suppressed_count)

    st.subheader("Anonymized Sample")
    display_cols = ['age', 'gender', 'department', 'salary']
    st.dataframe(anon_suppressed[display_cols].head(10))

    st.subheader("Group Size Distribution")
    group_sizes = anon_suppressed.groupby(QI).size().reset_index(
        name='group_size'
    )
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(group_sizes['group_size'], bins=20, color='#1D9E75', 
            edgecolor='white')
    ax.axvline(x=k_value, color='red', linestyle='--', 
               label=f'k={k_value}')
    ax.set_xlabel('Group Size')
    ax.set_ylabel('Number of Groups')
    ax.set_title('Distribution of K-Anonymous Group Sizes')
    ax.legend()
    st.pyplot(fig)