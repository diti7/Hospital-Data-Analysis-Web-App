import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import datetime

# Setting up the page config.
st.set_page_config(page_title = "Hospital Data Analysis", layout = "centered", page_icon = "📊")

# Title
st.title("🏥 Hospital Data Analysis")
st.markdown(""" This interactive tool allows you to explore hospital admission trends, patient demographics, and resource allocation.  
- 📊 **Compare** patient admissions across hospitals  
- 🔍 **Analyze** common conditions and average hospital stays  
- 📈 **Visualize** trends in hospital readmissions and severity levels  

👉 Select a dataset and generate insights!  
""")

working_dir = os.path.dirname(os.path.abspath(__file__))

# Datasets
folder_path = r"./Data"

# CSV Files
files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

# Dropdown menu
selected_file = st.selectbox("Select a file", files, index = None)
df = None

if selected_file:
    file_path = os.path.join(folder_path, selected_file)
    # Read the CSV file
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error loading file: {e}")

    if df is not None:

        # ------------------------------------ DATA CLEANING SECTION ------------------------------------
        st.subheader("🛠 Data Cleaning & Overview")
    # Data Cleaning -Expander
    with st.expander("🛠 Data Cleaning & Overview (Click to Expand)"):
        st.markdown("This section helps identify issues like missing values, duplicates, and incorrect data types. Use the options below to clean your dataset before visualization.")

        # Show missing values
        missing_values = df.isnull().sum()
        st.write("🔍 **Missing Values:**")
        st.write(missing_values[missing_values > 0])

        # Handling missing values
        clean_option = st.selectbox(
            "How do you want to handle missing values?",
            ["Do nothing", "Drop rows with missing values", "Fill with mean (numerical only)", "Fill with mode (categorical only)"]
        )

        if clean_option == "Drop rows with missing values":
            df.dropna(inplace=True)
            st.success("✅ Dropped rows with missing values!")

        elif clean_option == "Fill with mean (numerical only)":
            num_cols = df.select_dtypes(include=['number']).columns
            df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
            st.success("✅ Filled missing numerical values with mean!")

        elif clean_option == "Fill with mode (categorical only)":
            cat_cols = df.select_dtypes(include=['object']).columns
            df[cat_cols] = df[cat_cols].apply(lambda x: x.fillna(x.mode()[0]))
            st.success("✅ Filled missing categorical values with mode!")

        # Show duplicate rows
        duplicates = df.duplicated().sum()
        st.write(f"📌 **Duplicate Rows:** {duplicates}")

        # Handling duplicates
        if st.button("Remove Duplicate Rows"):
            df.drop_duplicates(inplace=True)
            st.success("✅ Removed duplicate rows!")

        # Show column data types
        st.write("📊 **Column Data Types:**")
        st.dataframe(df.dtypes.to_frame().T) #Transpose horizontally

        # Final Cleaned Data Preview!
        st.subheader("📌 Cleaned Data Preview")
        st.write(df.head())



    # ------------------------------------ ENHANCED ANALYTICS ------------------------------------
    st.subheader("📊 Enhanced Analytics & Data Visualization")

    # Summary Statistics - Expandable
    with st.expander("📈 View Summary Statistics (Click to Expand)"):
        st.write(df.describe().T)  # Transpose to display horizontally

    # Correlation Heatmap -Expandable
    if st.button("Show Correlation Heatmap"):
        # Selecting only numeric columns for correlation
        numeric_df = df.select_dtypes(include=['number'])
        
        if numeric_df.empty:
            st.error("❌ No numeric columns available for correlation.")
        else:
            fig_corr, ax_corr = plt.subplots(figsize=(8,6))
            sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr)
            st.pyplot(fig_corr)

    # Time Series Analysis (if "admission_date" exists)
    if "admission_date" in df.columns:
        df["admission_date"] = pd.to_datetime(df["admission_date"])  # Convert if needed
        df_time = df.groupby("admission_date")["admission_count"].sum()

        with st.expander("📅 View Time-Series Trend (Admissions Over Time)"):
            fig_time, ax_time = plt.subplots(figsize=(8,4))
            sns.lineplot(x=df_time.index, y=df_time.values, ax=ax_time)
            plt.xticks(rotation=45)
            plt.xlabel("Date")
            plt.ylabel("Total Admissions")
            plt.title("Admissions Over Time")
            st.pyplot(fig_time)

    # ------------------------------------ KEY INSIGHTS ------------------------------------
    # Check if there are insights to display
    key_insights_available = False
    # Total Admissions
    if "admission_count" in df.columns:
        total_admissions = df["admission_count"].sum()
        key_insights_available = True

    # Most Common Conditions
    if "condition_type" in df.columns:
        most_common_condition = df["condition_type"].value_counts().idxmax()
        key_insights_available = True

    # Hospital with most admissions
    if "hospital_name" in df.columns:
        top_hospital = df.groupby("hospital_name")["admission_count"].sum().idxmax()
        key_insights_available = True

    # Only to be displayed with there are insights!
    if key_insights_available:
        st.subheader("🔹 Key Insights")
        st.write(f"📌 **Total Admissions:** {total_admissions}")
        st.write(f"🩺 **Most Common Condition:** {most_common_condition}")
        st.write(f"🏥 **Hospital with the Most Admissions:** {top_hospital}")

    # ------------------------------------ MAIN SECTION ------------------------------------
    col1, col2 = st.columns(2)
    columns = df.columns.tolist()

    with col1:
        st.write("")
        st.write(df.head())

    with col2:
        # Selecting columns for plotting
        x_axis = st.selectbox("Select the X-axis", options = columns + ["None"])
        y_axis = st.selectbox("Select the Y-axis", options = columns + ["None"])

        # Selecting the type of plot
        plot_list = ["Line Plot", "Bar Chart", "Scatter Plot", "Distribution Plot", "Count Plot"]
        plot_type = st.selectbox("Select the type of plot", options = plot_list)

    # Generating the plot
    if st.button("Generate Plot"):
        fig, ax = plt.subplots(figsize = (6,4))
        
        if plot_type == "Line Plot":
            sns.lineplot(x = df[x_axis], y = df[y_axis], ax = ax)
        elif plot_type == "Bar Chart":
            sns.barplot(x = df[x_axis], y = df[y_axis], ax = ax)
        elif plot_type == "Scatter Plot":
            sns.scatterplot(x = df[x_axis], y = df[y_axis], ax = ax)
        elif plot_type == "Distribution Plot":
            sns.histplot(df[x_axis], kde = True, ax = ax)
            y_axis="Density"
        elif plot_type == "Count Plot":
            sns.countplot(x = df[x_axis], ax = ax)
            y_axis = "Count"

        # Adjusting label sizes
        ax.tick_params(axis = "x", labelsize = 10)
        ax.tick_params(axis = "y", labelsize = 10)

        #Adjusting title & axis labels font sizes
        plt.title(f'{plot_type} of {y_axis} vs {x_axis}', fontsize = 12)
        plt.xlabel(x_axis, fontsize = 10)
        plt.ylabel(y_axis, fontsize = 10)

        # Plot
        st.pyplot(fig)

    # ------------------------------------ FILTER DATA SECTION ------------------------------------
    st.sidebar.header("🔍 Filter Data")

    # generate filtering options for all columns!
    for column in df.columns:
        if df[column].dtype == 'object':  #categorical columns
            options = df[column].unique()
            selected_options = st.sidebar.multiselect(f"Select {column}", options=options)
            if selected_options:
                df = df[df[column].isin(selected_options)]
        
        elif df[column].dtype in ['int64', 'float64']:  #numerical columns
            min_value = df[column].min()
            max_value = df[column].max()
            selected_range = st.sidebar.slider(f"Select range for {column}", min_value, max_value, (min_value, max_value))
            df = df[df[column].between(selected_range[0], selected_range[1])]

    # Show the filtered dataset
    with st.expander("📋 View Filtered Data"):
        st.write(df.head())

    #------------------------------------ PREDICTIVE ADMISSIONS SECTION ------------------------------------
    st.subheader("📈 Predict Future Admissions")
    if "admission_count" in df.columns and "admission_date" in df.columns:
        #specific columnsx to Riyadh Hospital Admissions dataset!!!!
        df["admission_date"] = pd.to_datetime(df["admission_date"], errors='coerce')
        
        # drop rows with missing / invalid dates
        df = df.dropna(subset=["admission_date"])
        
        # Extracting year & month for regression model
        df["year"] = df["admission_date"].dt.year
        df["month"] = df["admission_date"].dt.month
        
        # preparing data for regression (predicting admission count based on year & month)
        X = df[["year", "month"]]
        y = df["admission_count"]
        
        # Training the regression model
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict future admissions (eg, for next 12 months)
        future_dates = pd.date_range(start=df["admission_date"].max(), periods=13, freq='M')
        future_X = pd.DataFrame({"year": future_dates.year, "month": future_dates.month})
        future_predictions = model.predict(future_X)
        
        # Display predictions!!!!
        st.write("📅 **Predicted Admissions for the Next 12 Months:**")
        future_predictions_df = pd.DataFrame({
            "Month": future_dates.month,
            "Year": future_dates.year,
            "Predicted Admissions": future_predictions and prospect
        })
        st.write(future_predictions_df)
    else:
        st.write("❌ **Insufficient data for prediction.**")
