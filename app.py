import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="Loan Data EDA Dashboard", layout="wide")

sns.set_theme(style="whitegrid")

# Load and cache data
@st.cache_data
def load_data():
    df = pd.read_csv("loan_data_clean.csv")
    
    # Feature engineering for missing/derived columns
    if "APPROVED_FLAG" not in df.columns:
        df["APPROVED_FLAG"] = (df["NAME_CONTRACT_STATUS"] == "Approved").astype(int)
        
    if "YEARS_SINCE_DECISION" not in df.columns and "DAYS_DECISION" in df.columns:
        df["YEARS_SINCE_DECISION"] = df["DAYS_DECISION"].abs() / 365.0
        
    for col in ["AMT_CREDIT", "AMT_APPLICATION"]:
        if col in df.columns and f"{col}_CAPPED" not in df.columns:
            cap_value = df[col].quantile(0.99)
            df[f"{col}_CAPPED"] = df[col].clip(upper=cap_value)
            
    return df

df = load_data()

# App Sidebar Navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Choose Analysis Section", [
    "1. Overview", 
    "2. Numeric Distributions", 
    "3. Categorical Breakdown", 
    "4. Approval Rates", 
    "5. Rejection Reasons", 
    "6. Trends Over Time"
])

st.title("📊 Loan Data Exploratory Data Analysis")

# --- SECTION 1: OVERVIEW ---
if section == "1. Overview":
    st.header("Dataset Overview")
    st.write(f"*Dataset Shape:* {df.shape[0]} rows, {df.shape[1]} columns")
    
    st.subheader("Data Sample")
    st.dataframe(df.head())
    
    st.subheader("Statistical Summary")
    st.dataframe(df.describe())

# --- SECTION 2: NUMERIC DISTRIBUTIONS ---
elif section == "2. Numeric Distributions":
    st.header("Univariate Analysis: Numeric Features")
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    cols = [c for c in ["AMT_CREDIT_CAPPED", "AMT_APPLICATION_CAPPED"] if c in df.columns]
    for a, c in zip(ax, cols):
        sns.histplot(df[c].dropna(), bins=40, kde=True, ax=a)
        a.set_title(c)
    st.pyplot(fig)
    st.caption("Insight: Capping extreme values normalizes the distributions for analysis.")

    st.subheader("AMT_CREDIT Outliers Before Capping")
    fig2, ax2 = plt.subplots(figsize=(8, 3))
    sns.boxplot(x=df["AMT_CREDIT"], ax=ax2)
    ax2.set_title("AMT_CREDIT with outliers")
    st.pyplot(fig2)
    st.caption("Insight: The boxplot illustrates the heavy right-skew and significant outliers present in raw credit amounts.")

# --- SECTION 3: CATEGORICAL BREAKDOWN ---
elif section == "3. Categorical Breakdown":
    st.header("Univariate Analysis: Categorical Features")
    
    cat_col = st.selectbox("Select Categorical Column", [
        "NAME_CONTRACT_STATUS", "NAME_CONTRACT_TYPE", "NAME_CLIENT_TYPE",
        "NAME_PORTFOLIO", "CHANNEL_TYPE", "NAME_PAYMENT_TYPE"
    ])
    
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(data=df, y=cat_col, order=df[cat_col].value_counts().index, ax=ax)
    ax.set_title(f"Distribution of {cat_col}")
    st.pyplot(fig)
    st.caption(f"Insight: Frequency counts highlight dominant categories within {cat_col}.")

# --- SECTION 4: APPROVAL RATES ---
elif section == "4. Approval Rates":
    st.header("Bivariate Analysis: Approval Rate by Segment")
    
    seg = st.selectbox("Select Segment", ["NAME_CONTRACT_TYPE", "NAME_CLIENT_TYPE", "NAME_PORTFOLIO", "CHANNEL_TYPE"])
    
    if "APPROVED_FLAG" in df.columns:
        rate = (df.groupby(seg)["APPROVED_FLAG"].mean() * 100).round(2).sort_values()
        
        fig, ax = plt.subplots(figsize=(8, 4))
        rate.plot(kind="barh", ax=ax, color="teal")
        ax.set_title(f"Approval % by {seg}")
        st.pyplot(fig)
        st.caption("Insight: Approval rates fluctuate significantly across different segments.")

# --- SECTION 5: REJECTION REASONS ---
elif section == "5. Rejection Reasons":
    st.header("Why Applications Get Refused")
    
    if "CODE_REJECT_REASON" in df.columns:
        ref = df[df["NAME_CONTRACT_STATUS"] == "Refused"]
        
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(data=ref, y="CODE_REJECT_REASON", order=ref["CODE_REJECT_REASON"].value_counts().index, ax=ax)
        ax.set_title("Reject Reasons")
        st.pyplot(fig)
        st.caption("Insight: A vast majority of loan refusals concentrate within a few primary rejection codes.")

# --- SECTION 6: TRENDS OVER TIME ---
elif section == "6. Trends Over Time":
    st.header("Approval Trend Over Time")
    
    if "YEARS_SINCE_DECISION" in df.columns and "APPROVED_FLAG" in df.columns:
        trend = df.groupby(df["YEARS_SINCE_DECISION"].round())["APPROVED_FLAG"].mean() * 100
        
        fig, ax = plt.subplots(figsize=(8, 4))
        trend.plot(marker="o", ax=ax, color="purple")
        ax.set_title("Approval % by Years Since Decision")
        ax.set_ylabel("Approval Rate (%)")
        st.pyplot(fig)
        st.caption("Insight: Historical trend indicates shifting approval rates over time.")

