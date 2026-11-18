import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from urllib.parse import quote_plus
from sqlalchemy import create_engine

# --- DATABASE CONFIGURATION ---
DB_USER = "root"
DB_PASSWORD = "1234"  # Update with your MySQL root password
DB_HOST = "localhost"
DB_PORT = 3306
DB_NAME = "loan_analysis"

# Set page config
st.set_page_config(page_title="loan_analysis & Loan EDA Dashboard", layout="wide")
sns.set_theme(style="whitegrid")

# --- DATABASE & CSV ENGINE ---
@st.cache_resource
def get_engine():
    url = f"mysql+pymysql://{DB_USER}:{quote_plus(DB_PASSWORD)}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(url)

@st.cache_data
def run_sql(query):
    return pd.read_sql(query, get_engine())

@st.cache_data
def load_csv_data():
    try:
        df = pd.read_csv("loan_data_clean.csv")
    except FileNotFoundError:
        # Fallback if csv isn't found locally, fetch from MySQL table
        df = run_sql("SELECT * FROM loan_data")
        
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

df = load_csv_data()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
section = st.sidebar.radio("Choose Section", [
    "📊 EDA & Visualizations", 
    "🗄️ SQL Queries & Database Reports"
])

# ==========================================
# SECTION 1: EDA & VISUALIZATIONS
# ==========================================
if section == "📊 EDA & Visualizations":
    st.title("📊 Loan Data Exploratory Data Analysis & Dashboard")
    
    eda_tab = st.sidebar.selectbox("EDA Sub-Section", [
        "1. Overview", 
        "2. Numeric Distributions", 
        "3. Categorical Breakdown", 
        "4. Approval Rates", 
        "5. Rejection Reasons", 
        "6. Trends Over Time"
    ])

    if eda_tab == "1. Overview":
        st.header("Dataset Overview")
        st.write(f"*Dataset Shape:* {df.shape[0]} rows, {df.shape[1]} columns")
        
        # High-level KPIs from MySQL / CSV
        try:
            kpis = run_sql("""
                SELECT COUNT(*) AS total,
                       SUM(AMT_CREDIT) AS credit,
                       AVG(AMT_CREDIT) AS avg_credit,
                       100 * AVG(NAME_CONTRACT_STATUS = 'Approved') AS approval_pct
                FROM loan_data
            """).iloc[0]
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Applications", f"{int(kpis['total']):,}")
            c2.metric("Total Credit", f"{float(kpis['credit']) / 1e9:,.2f}B")
            c3.metric("Avg Credit", f"{float(kpis['avg_credit']):,.0f}")
            c4.metric("Approval Rate", f"{float(kpis['approval_pct']):.1f}%")
        except Exception:
            pass

        st.subheader("Data Sample")
        st.dataframe(df.head())
        
        st.subheader("Statistical Summary")
        st.dataframe(df.describe())

    elif eda_tab == "2. Numeric Distributions":
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

    elif eda_tab == "3. Categorical Breakdown":
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

    elif eda_tab == "4. Approval Rates":
        st.header("Bivariate Analysis: Approval Rate by Segment")
        
        seg = st.selectbox("Select Segment", ["NAME_CONTRACT_TYPE", "NAME_CLIENT_TYPE", "NAME_PORTFOLIO", "CHANNEL_TYPE"])
        
        if "APPROVED_FLAG" in df.columns:
            rate = (df.groupby(seg)["APPROVED_FLAG"].mean() * 100).round(2).sort_values()
            
            fig, ax = plt.subplots(figsize=(8, 4))
            rate.plot(kind="barh", ax=ax, color="teal")
            ax.set_title(f"Approval % by {seg}")
            st.pyplot(fig)
            st.caption("Insight: Approval rates fluctuate significantly across different segments.")

    elif eda_tab == "5. Rejection Reasons":
        st.header("Why Applications Get Refused")
        
        if "CODE_REJECT_REASON" in df.columns:
            ref = df[df["NAME_CONTRACT_STATUS"] == "Refused"]
            
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.countplot(data=ref, y="CODE_REJECT_REASON", order=ref["CODE_REJECT_REASON"].value_counts().index, ax=ax)
            ax.set_title("Reject Reasons")
            st.pyplot(fig)
            st.caption("Insight: A vast majority of loan refusals concentrate within a few primary rejection codes.")

    elif eda_tab == "6. Trends Over Time":
        st.header("Approval Trend Over Time")
        
        if "YEARS_SINCE_DECISION" in df.columns and "APPROVED_FLAG" in df.columns:
            trend = df.groupby(df["YEARS_SINCE_DECISION"].round())["APPROVED_FLAG"].mean() * 100
            
            fig, ax = plt.subplots(figsize=(8, 4))
            trend.plot(marker="o", ax=ax, color="purple")
            ax.set_title("Approval % by Years Since Decision")
            ax.set_ylabel("Approval Rate (%)")
            st.pyplot(fig)
            st.caption("Insight: Historical trend indicates shifting approval rates over time.")

# ==========================================
# SECTION 2: SQL QUERIES & DATABASE REPORTS
# ==========================================
elif section == "🗄️ SQL Queries & Database Reports":
    st.title("🗄️ Mandatory SQL Analytical Queries")
    st.caption("Execute all 20 mandatory SQL queries directly against your MySQL database.")

    QUERIES = {
        "Q1  All records (first 1000)": "SELECT * FROM loan_data LIMIT 1000",
        "Q2  Contract type and status (first 1000)": "SELECT SK_ID_CURR, NAME_CONTRACT_TYPE, NAME_CONTRACT_STATUS FROM loan_data LIMIT 1000",
        "Q3  Approved applications (first 1000)": "SELECT * FROM loan_data WHERE NAME_CONTRACT_STATUS = 'Approved' LIMIT 1000",
        "Q4  Refused applications (first 1000)": "SELECT * FROM loan_data WHERE NAME_CONTRACT_STATUS = 'Refused' LIMIT 1000",
        "Q5  Repeater clients (first 1000)": "SELECT * FROM loan_data WHERE NAME_CLIENT_TYPE = 'Repeater' LIMIT 1000",
        "Q6  Total applications": "SELECT COUNT(*) AS total_applications FROM loan_data",
        "Q7  Total credit amount": "SELECT ROUND(SUM(AMT_CREDIT), 2) AS total_credit FROM loan_data",
        "Q8  Average application amount": "SELECT ROUND(AVG(AMT_APPLICATION), 2) AS avg_application_amount FROM loan_data",
        "Q9  Applications per status": "SELECT NAME_CONTRACT_STATUS, COUNT(*) AS applications FROM loan_data GROUP BY NAME_CONTRACT_STATUS",
        "Q10 Average annuity per contract type": "SELECT NAME_CONTRACT_TYPE, ROUND(AVG(AMT_ANNUITY), 2) AS avg_annuity FROM loan_data GROUP BY NAME_CONTRACT_TYPE",
        "Q11 Credit above 500,000 (first 1000)": "SELECT * FROM loan_data WHERE AMT_CREDIT > 500000 LIMIT 1000",
        "Q12 Down payment above 50,000 (first 1000)": "SELECT * FROM loan_data WHERE AMT_DOWN_PAYMENT > 50000 LIMIT 1000",
        "Q13 Applications per client type": "SELECT NAME_CLIENT_TYPE, COUNT(*) AS applications FROM loan_data GROUP BY NAME_CLIENT_TYPE ORDER BY applications DESC",
        "Q14 Average credit per status": "SELECT NAME_CONTRACT_STATUS, ROUND(AVG(AMT_CREDIT), 2) AS avg_credit FROM loan_data GROUP BY NAME_CONTRACT_STATUS",
        "Q15 Top 5 product combinations": "SELECT PRODUCT_COMBINATION, COUNT(*) AS applications FROM loan_data GROUP BY PRODUCT_COMBINATION ORDER BY applications DESC LIMIT 5",
        "Q16 Credit: approved vs refused": "SELECT NAME_CONTRACT_STATUS, ROUND(SUM(AMT_CREDIT), 2) AS total_credit FROM loan_data WHERE NAME_CONTRACT_STATUS IN ('Approved', 'Refused') GROUP BY NAME_CONTRACT_STATUS",
        "Q17 Top 3 contract types by avg credit": "SELECT NAME_CONTRACT_TYPE, ROUND(AVG(AMT_CREDIT), 2) AS avg_credit FROM loan_data GROUP BY NAME_CONTRACT_TYPE ORDER BY avg_credit DESC LIMIT 3",
        "Q18 Application amount per payment type": "SELECT NAME_PAYMENT_TYPE, ROUND(SUM(AMT_APPLICATION), 2) AS total_application_amount FROM loan_data GROUP BY NAME_PAYMENT_TYPE ORDER BY total_application_amount DESC",
        "Q19 Top 5 seller industries (with Unknown)": "SELECT NAME_SELLER_INDUSTRY, ROUND(SUM(AMT_CREDIT), 2) AS total_credit FROM loan_data GROUP BY NAME_SELLER_INDUSTRY ORDER BY total_credit DESC LIMIT 5",
        "Q19b Top 5 seller industries (without Unknown)": "SELECT NAME_SELLER_INDUSTRY, ROUND(SUM(AMT_CREDIT), 2) AS total_credit FROM loan_data WHERE NAME_SELLER_INDUSTRY <> 'Unknown' GROUP BY NAME_SELLER_INDUSTRY ORDER BY total_credit DESC LIMIT 5",
        "Q20 Approval % per contract type": "SELECT NAME_CONTRACT_TYPE, ROUND(100.0 * SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Approved' THEN 1 ELSE 0 END) / COUNT(*), 2) AS approval_pct FROM loan_data GROUP BY NAME_CONTRACT_TYPE",
    }

    choice = st.selectbox("Choose a mandatory SQL query", list(QUERIES.keys()))
    st.code(QUERIES[choice], language="sql")
    
    if st.button("Run SQL Query"):
        try:
            res_df = run_sql(QUERIES[choice])
            st.success(f"Query executed successfully! Returned {len(res_df):,} rows.")
            st.dataframe(res_df)
        except Exception as e:
            st.error(f"Error executing query: {e}. Please ensure MySQL is running and DB_PASSWORD is set correctly.")