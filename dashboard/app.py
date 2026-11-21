
import streamlit as st
import pandas as pd
import plotly.express as px

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Financial Anomaly Detection",
    page_icon="💳",
    layout="wide"
)

# -------------------------------
# Dashboard Header
# -------------------------------
st.title("💳 Financial Anomaly Detection Dashboard")
st.markdown("### Statistical Machine Learning Based Fraud Analysis")
st.markdown("---")

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Overview", "Data Analysis"]
)

# -------------------------------
# Data Upload
# -------------------------------
st.sidebar.subheader("Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.success("Dataset uploaded successfully!")

    # -------------------------------
    # Dataset Overview
    # -------------------------------
    if page == "Overview":

        st.subheader("Dataset Overview")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Transactions", f"{len(df):,}")

        with col2:
            st.metric("Total Features", len(df.columns))

        with col3:
            st.metric("Missing Values", f"{df.isnull().sum().sum():,}")

        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)

    # -------------------------------
    # Data Analysis
    # -------------------------------
    elif page == "Data Analysis":

        st.subheader("Data Analysis")

        st.write("### Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)

        numeric_columns = df.select_dtypes(include="number").columns

        if len(numeric_columns) > 0:

            selected_column = st.selectbox(
                "Select feature to visualize",
                numeric_columns
            )

            fig = px.histogram(
                df,
                x=selected_column,
                title=f"Distribution of {selected_column}"
            )

            st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Upload a CSV dataset from the sidebar to get started.")

    st.markdown("""
    ### Features
    - Dataset overview
    - Transaction statistics
    - Interactive data visualization
    - Feature distribution analysis
    """)
