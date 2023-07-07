import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.ensemble import IsolationForest
import scipy.stats as stats
from sklearn.neighbors import KernelDensity
from sklearn.cluster import KMeans
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import norm
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import plotly.express as px

import base64
import warnings
import webbrowser

import streamlit as st

from matplotlib import style
style.use("ggplot")
st.set_option('deprecation.showPyplotGlobalUse', False)

# Define the HTML code with CSS for the marquee

# Add custom CSS
st.markdown(
    """
    <style>
    body {
        background-color: #ffffff; /* Set the background color */
        color: #222222; /* Set the text color */
        font-weight: bold; /* Set the text weight */
        font-family: sans-serif; /* Set the font family */
    }
    </style>
    """,
    unsafe_allow_html=True
)

marquee_html = """
<html>
<head>
  <title>Beautify Image</title>
  <style>
    .heading {
      font-size: 2rem;
      color: red;
      text-shadow: 2px 2px 2px #000;
    }

    .marquee {
      width: 100%;
      overflow: hidden;
      background-color: rgb(80 175 183);
      color: #fff;
      padding: 20px;
      margin-top: 0; /* Adjust the margin-top value here */
      text-align: center;
    }

    .marquee span {
      display: inline-block;
      animation: marquee 20s linear infinite;
      color: #111;
      background-color: #fff;
      font-size: 1.2rem;
      width: 2000px; /* Slim the span */
      /* height: 50px; Slim the span in height */
      max-height: 40px; /* This will make the span 50px tall at most */
      
  

    }

    .logo {
      width: 150px;
      position: relative;
      bottom: 20px;
    }

    .P_logo {
      width: 300px;
      position: relative;
      bottom: 20px;
    }

    h1 {
      font-size: 60px;
      text-align: left;
      position: relative;
      top: 100px;
    }

    @keyframes marquee {
      0% { transform: translateX(100%); }
      100% { transform: translateX(-100%); }
    }
  </style>
</head>
<body>
  <div id="right" style="display: flex;justify-content: space-between;">
    <img src="https://i.ibb.co/VQCBVxg/templogo.jpg" class="P_logo" style="max-width: 70%; height: auto;">
    <img src="https://revoquant.com/assets/img/logo/logo-dark.png" class="logo" style="max-width: 100%; height: auto;">
  </div>
  <div class="marquee">
    <span style="color: #000; background-color: #fff;">
This application provides an advanced method for understanding your dataset and detecting outliers. It comes with pre-built statistical and machine learning models specifically designed to identify outliers in large-scale data.    </span>
  </div>
  <center>
    <img src="https://github.com/MANMEET75/INFRARED/raw/main/infrared1.gif" alt="GIF" style="max-width: 100%; height: auto; width: 150%; height: 350px;">
  </center>
</body>
</html>

"""

# Render the marquee in Streamlit
st.markdown(marquee_html, unsafe_allow_html=True)

def get_download_link(file_path):
    with open(file_path, "rb") as file:
        contents = file.read()
    encoded_file = base64.b64encode(contents).decode("utf-8")
    href = f'<a href="data:file/csv;base64,{encoded_file}" download="{file_path}">Click here to download</a>'
    return href

def drop_features_with_missing_values(data):
    # Calculate the number of missing values in each column
    missing_counts = data.isnull().sum()

    # Get the names of columns with missing values
    columns_with_missing_values = missing_counts[missing_counts > 0].index

    # Drop the columns with missing values
    data_dropped = data.drop(columns=columns_with_missing_values)
    return data_dropped





def apply_anomaly_detection_IsolationForest(data):
    # Make a copy of the data
    data_copy = data.copy()

    # Fit the Isolation Forest model
    isolation_forest = IsolationForest(contamination=0.05, random_state=42)
    isolation_forest.fit(data_copy)

    # Predict the anomaly labels
    anomaly_labels = isolation_forest.predict(data_copy)

    # Create a new column in the original DataFrame for the anomaly indicator
    data['Anomaly'] = np.where(anomaly_labels == -1, 1, 0)
    return data





def find_duplicate_vendors(vendors_df, threshold):
    duplicates = []
    lf = vendors_df.copy()
    lf['NAME1'] = lf['NAME1'].astype(str)
    vendor_names = lf['NAME1'].unique().tolist()
    columns = ['Vendor 1', 'Vendor 2', 'Score']
    df_duplicates = pd.DataFrame(data=[], columns=columns)

    for i, name in enumerate(vendor_names):
        # Compare the current name with the remaining names
        matches = process.extract(name, vendor_names[i+1:], scorer=fuzz.ratio)

        # Check if any match exceeds the threshold
        for match, score in matches:
            if score >= threshold:
                duplicates.append((name, match))
                df_duplicates.loc[len(df_duplicates)] = [name, match, score]

    return duplicates, df_duplicates


def calculate_first_digit(data):
    first_digits = data.astype(str).str.strip().str[0].astype(int)
    counts = first_digits.value_counts(normalize=True, sort=False)
    benford = np.log10(1 + 1 / np.arange(0, 10))
    return counts, benford
def calculate_2th_digit(data):
    nth_digits = data.astype(int).astype(str).str.strip().str[:2]
    numeric_mask = nth_digits.str.isnumeric()
    counts = nth_digits[numeric_mask].astype(int).value_counts(normalize=True, sort=False)
    benford = np.log10(1 + 1 / np.arange(0, 100))
    return counts, benford

def calculate_3th_digit(data):
    nth_digits = data.astype(int).astype(str).str.strip().str[:3]
    numeric_mask = nth_digits.str.isnumeric()
    counts = nth_digits[numeric_mask].astype(int).value_counts(normalize=True, sort=False)
    benford = np.log10(1 + 1 / np.arange(0, 1000))
    return counts, benford
def main():




    st.header("Upload your data file")
    data_file = st.file_uploader("Upload File", type=["csv", "xlsx", "XLSX"])

    if data_file is not None:
        file_extension = data_file.name.split(".")[-1]
        if file_extension == "csv":
            data = pd.read_csv(data_file)
        elif file_extension in ["xlsx", "XLSX"]:
            data = pd.read_excel(data_file)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")


        # starting of basic information side bar
        st.sidebar.header("Exploratory Data Analysis")
        info_options = ["None",
                        "Number of one time vendor account",
                        "Duplicate vendor account basis name and fuzzy logic",
                        "Changes in vendor bank account",
                        "Vendor details matching with employee (common address, common bank account, common PAN)",
                        "Dormant vendor",
                        "Vendor bank account is in different country or city than the country or city registered with",
                        "Vendor belongs to country in Africa, Turkey, Bangkok etc",
                        "Vendor having email address domain as gmail, yahoomail, Hotmail, rediffmail etc",
                        "Vendor is not private limited or public limited or LLP",
                        "Vendor who is having maximum quality rejections",
                        "Vendor who is having tendency to do delayed delivery. This could result in short supply",
                        "Vendor who is having tendency to delivery well in advance. This raises risk of build up inventory",
                        "Vendor who is having tendency to deliver the goods on same date of PO or nearby date of PO",
                        "Vendor with transactions of return PO using 161 mvt type and for those specific material-vendor combination, the avg return price is lower than the procurement price",
                        "Vendor where quality rejection takes significant time and the invoice is already paid or posted",
                        "Vendor which gets considered during invoice posting but not at the time of PO. This means, in PO vendor is showing X whereas in invoice posting vendor is showing as Y",
        ]
        selected_info = st.sidebar.selectbox("Choose an EDA type", info_options)

        if selected_info == "None":
            st.write(" ")
        elif selected_info == "Number of one time vendor account":
            st.header("Upload your lfa1 file")
            data_file1 = st.file_uploader("Upload CSV", type=["csv"], key="lfa1")

            if data_file1 is not None:
                data1 = pd.read_csv(data_file1, encoding='latin1')  # Specify the correct encoding
                st.write(data1.head(2))

            st.header("Upload your lfb1 file")
            data_file2 = st.file_uploader("Upload CSV", type=["csv"], key="lfb1")

            if data_file2 is not None:
                data2 = pd.read_csv(data_file2, encoding='latin1')  # Specify the correct encoding
                st.write(data2.head(2))

                # Merge data1 and data2 based on LIFNR using inner join
                merged_data = pd.merge(data1, data2, on="LIFNR", how="inner")
                st.write(merged_data.head(2))

                # Calculate the total number of unique vendors from 'NAME1' feature
                unique_vendors = merged_data['NAME1'].nunique()
                st.write("Total number of unique vendors:", unique_vendors)

                # Filter merged_data for rows where NAME1 occurs only once
                filtered_x2 = merged_data[merged_data['NAME1'].map(merged_data['NAME1'].value_counts()) == 1]
                st.write(filtered_x2)

                # Filter merged_data for rows where XCPDK is equal to 'X'
                one_time_vendor = merged_data[merged_data['XCPDK'] == 'X']
                st.write(one_time_vendor)

        elif selected_info == "Duplicate vendor account basis name and fuzzy logic":
            st.header("Upload your lfa1 file")
            data_file1 = st.file_uploader("Upload CSV", type=["csv"], key="lfa1")

            if data_file1 is not None:
                data1 = pd.read_csv(data_file1, encoding='latin1')  # Specify the correct encoding
                st.write(data1.head(2))

                # Perform duplicate vendor detection
                threshold = 95
                duplicates, df_duplicates = find_duplicate_vendors(data1, threshold)

                # Display duplicate vendor pairs
                st.write("Duplicate Vendor Accounts:")
                for vendor1, vendor2 in duplicates:
                    st.write(vendor1, "--", vendor2)

                # Display dataframe with duplicate vendor pairs
                st.write("Duplicate Vendor Accounts DataFrame:")
                st.write(df_duplicates)

        elif selected_info == "Changes in vendor bank account":
            st.header("Upload your lfbk file")
            data_file1 = st.file_uploader("Upload CSV", type=["csv"], key="lfbk")

            if data_file1 is not None:
                data1 = pd.read_csv(data_file1, encoding='latin1')  # Specify the correct encoding
                st.write(data1.head(2))

                # Calculate the count of unique BANKN values for each LIFNR
                series_data = data1.groupby('LIFNR')['BANKN'].nunique()

                # Create a DataFrame with the count of BANKN values
                x1 = pd.DataFrame({'BANKN': series_data})
                x1 = x1.rename(columns={'BANKN': 'count_BankN'})
                x1 = x1.reset_index()

                # Get the list of LIFNR values with count_BankN greater than 1
                l1 = list(x1[x1['count_BankN'] > 1]['LIFNR'])

                # Filter x1 for rows where count_BankN is greater than 1
                x1_filtered = x1[x1['count_BankN'] > 1]

                # Count the number of people with multiple accounts
                num_people = len(x1_filtered)

                # Display the number of people with multiple accounts
                st.write("Number of people with multiple accounts:", num_people)
                st.write(x1_filtered)

        elif selected_info == "Vendor details matching with employee (common address, common bank account, common PAN)":
            st.header("Upload your lfa1 file")
            data_file1 = st.file_uploader("Upload CSV", type=["csv"], key="lfa1")
            data1 = None  # Initialize data1 variable
            data2 = None  # Initialize data2 variable
            data3 = None  # Initialize data3 variable
            data4 = None  # Initialize data4 variable

            if data_file1 is not None:
                data1 = pd.read_csv(data_file1, encoding='latin1')  # Specify the correct encoding
                data1['KTOKK'] = data1['KTOKK'].astype(str)  # Convert 'KTOKK' column to string
                data1['KTOKK'] = data1['KTOKK'].str.replace("'", "")  # Replace single quotes in 'KTOKK' column
                st.write(data1.head(2))

            st.header("Upload your Vendor_acc file")
            data_file2 = st.file_uploader("Upload CSV", type=["csv"], key="Vendor_acc")

            if data_file2 is not None:
                data2 = pd.read_csv(data_file2, encoding='latin1')  # Specify the correct encoding
                data2['Group'] = data2['Group'].astype(str)  # Convert 'Group' column to string
                st.write(data2.head(2))

            if data1 is not None and data2 is not None:
                lfa1_vd = pd.merge(data1, data2, left_on='KTOKK', right_on='Group', how='left')
                st.write(lfa1_vd)

                st.header("Upload your j_1 file")
                data_file3 = st.file_uploader("Upload CSV", type=["csv"], key="j_1")

                if data_file3 is not None:
                    data3 = pd.read_csv(data_file3, encoding='latin1')  # Specify the correct encoding
                    st.write(data3.head(2))

                st.header("Upload your lfbk file")
                data_file4 = st.file_uploader("Upload CSV", type=["csv"], key="lfbk")

                if data_file4 is not None:
                    data4 = pd.read_csv(data_file4, encoding='latin1')  # Specify the correct encoding
                    st.write(data4.head(2))

                if data3 is not None:
                    lfa1_vd_j = pd.merge(lfa1_vd, data3, on='LIFNR', how='left')

                    lfa1_vd_j_lfbk = pd.merge(lfa1_vd_j, data4, on='LIFNR', how='left')

                    lfa1_vd_j1 = lfa1_vd_j_lfbk[['LIFNR', 'LAND1', 'NAME1', 'ORT01', 'STRAS', 'STCD3', 'Name', 'BANKN', 'J_1IPANNO_x', 'J_1IPANNO_y']]
                    emp = lfa1_vd_j1[lfa1_vd_j1['Name'].str.contains('Empl')]
                    vendor = lfa1_vd_j1[~(lfa1_vd_j1['Name'].str.contains('Empl'))]

                    table1_bankn_set = set(vendor['BANKN'])
                    employee_bankn_set = set(emp['BANKN'])

                    common_bankn = table1_bankn_set.intersection(employee_bankn_set)
                    common_bankn_count = len(common_bankn)
                    emp[['LIFNR','BANKN']].dropna(inplace=True)
                    st.write(f"Total number of persons having a common bank account: {common_bankn_count}")
                    # for bankn in common_bankn:
                    #     st.write(bankn)

                    table1_J_1IPANNO_x_set = set(vendor['J_1IPANNO_x'])
                    employee_J_1IPANNO_x_set = set(emp['J_1IPANNO_x'])

                    common_J_1IPANNO_x = table1_J_1IPANNO_x_set.intersection(employee_J_1IPANNO_x_set)
                    common_J_1IPANNO_x_count = len(common_J_1IPANNO_x)
                    st.write(f"Total number of persons having a common pan: {common_J_1IPANNO_x_count}")

                    # for pan in common_J_1IPANNO_x:
                    #     st.write(pan)


                    table1_bankn_set = set(vendor['STRAS'])
                    employee_bankn_set = set(emp['STRAS'])

                    common_J_1IPANNO_xa = table1_bankn_set.intersection(employee_bankn_set)
                    common_J_1IPANNO_xa_count =len(common_J_1IPANNO_xa)
                    st.write(f"Total number of persons having a common address: {common_J_1IPANNO_xa_count}")

                    # for bankn in common_J_1IPANNO_x:
                    #     print(bankn)
                else:
                    st.write("Please upload the j_1 file.")
            else:
                st.write("Please upload both lfa1 and Vendor_acc files.")

        elif selected_info == "Vendor belongs to country in Africa, Turkey, Bangkok etc":
            st.header("Upload your lfa1 file")
            data_file1 = st.file_uploader("Upload CSV", type=["csv"], key="lfa1")
            if data_file1 is not None:
                data1 = pd.read_csv(data_file1, encoding='latin1')  # Specify the correct encoding
                st.write(data1.head(2))

            st.header("Upload your t005t file")
            data_file2 = st.file_uploader("Upload CSV", type=["csv"], key="t005t")

            if data_file2 is not None:
                data2 = pd.read_csv(data_file2, encoding='latin1')  # Specify the correct encoding
                st.write(data2.head(2))

                t005t = data2  # Assign data2 to t005t variable
                ltf = pd.merge(data1, t005t, on='LAND1', how='left')
                st.write(ltf)

                ltf1 = ltf[['LIFNR','LAND1', 'LANDX50']]
                filter_values = ['Africa', 'Turkey', 'Bangkok']  # Example filter values
                filter_country = ltf1[ltf1['LANDX50'].isin(filter_values)]
                country_counts = filter_country['LANDX50'].value_counts()
                st.write(country_counts.head(5))






























        # # Starting adding the feature engineering part over here separte options
        # st.sidebar.header("Feature Engineering")
        # info_options = [
        #     "None",
        #     "Deal with missing values",
        #     "Deal with duplicate values if present",
        #     "Encoding the categorical features",
        #     "Feature Scaling",
        #     "Download your dataset"
        # ]

        # selected_info = st.sidebar.selectbox("Perform feature engineering", info_options)

        # if selected_info == "None":
        #     st.write(" ")
        # else:
        #     if selected_info == "Deal with missing values":
        #         st.write("Please select a method to deal with missing values:")
        #         method_options = ["Drop missing values", "Fill missing values"]
        #         selected_method = st.selectbox("Method", method_options)

        #         if selected_method == "None":
        #             st.write("")
        #         elif selected_method == "Drop missing values":
        #             threshold = 0.1  # Set the threshold to 10% (0.1)
        #             missing_percentages = data.isnull().mean()  # Calculate the percentage of missing values in each column
        #             columns_to_drop = missing_percentages[missing_percentages > threshold].index  # Get the columns exceeding the threshold
        #             data.drop(columns=columns_to_drop, inplace=True)  # Drop the columns
        #             st.write(f"Features with more than {threshold*100:.2f}% missing values dropped successfully.")
        #         elif selected_method == "Fill missing values":
        #             # Your code for filling missing values goes here
        #             st.write("Missing values filled successfully.")

        #     if selected_info == "Deal with duplicate values if present":
        #         st.write("Dealing with duplicate values...")

        #         num_duplicates = data.duplicated().sum()  # Count the number of duplicate rows

        #         data.drop_duplicates(inplace=True)  # Drop the duplicate rows

        #         st.write(f"Number of duplicate rows: {num_duplicates}")
        #         st.write("Duplicate values removed successfully.")

        #     if selected_info == "Encoding the categorical features":
        #         st.write("Performing categorical feature encoding...")

        #         categorical_features = [feature for feature in data.columns if data[feature].dtype == 'object']

        #         for feature in categorical_features:
        #             labels_ordered = data.groupby([feature]).size().sort_values().index
        #             labels_ordered = {k: i for i, k in enumerate(labels_ordered, 0)}
        #             data[feature] = data[feature].map(labels_ordered)

        #         st.write("Categorical features encoded successfully.")

        #     if selected_info == "Feature Scaling":
        #         st.write("Performing feature scaling...")

        #         # Identify numeric columns
        #         numeric_columns = data.select_dtypes(include=["int", "float"]).columns

        #         if len(numeric_columns) == 0:
        #             st.write("No numeric columns found.")
        #         else:
        #             scaler = MinMaxScaler()
        #             data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

        #         st.write("Feature scaling performed successfully.")

        #     if selected_info == "Download your dataset":
        #         st.write("Downloading the dataset...")

        #         # Perform any necessary feature engineering operations

        #         # Save the modified dataset to a file
        #         modified_dataset_filename = "modified_dataset.csv"
        #         data.to_csv(modified_dataset_filename, index=False)
        #         st.write(data.head())


        # Starting adding the feature engineering part over here

        st.sidebar.header("Feature Engineering")
        info_options = [
            "None",
            "Perform feature engineering",
        ]

        selected_info = st.sidebar.selectbox("Choose the option", info_options)

        if selected_info == "None":
            st.write(" ")
        else:
            if selected_info == "Perform feature engineering":

                st.write("Dealing with missing values:")
                threshold = 0.1  # Set the threshold to 10% (0.1)
                missing_percentages = data.isnull().mean()  # Calculate the percentage of missing values in each column
                columns_to_drop = missing_percentages[missing_percentages > threshold].index  # Get the columns exceeding the threshold
                data = data.drop(columns=columns_to_drop)  # Drop the columns
                st.write(f"Features with more than {threshold*100:.2f}% missing values dropped successfully.")



                data = drop_features_with_missing_values(data)




                st.write("Dealing with duplicate values...")
                num_duplicates = data.duplicated().sum()  # Count the number of duplicate rows
                data_unique = data.drop_duplicates()  # Drop the duplicate rows
                st.write(f"Number of duplicate rows: {num_duplicates}")
                st.write("Dealing done with duplicates.")

                st.write("Performing categorical feature encoding...")
                categorical_features = [feature for feature in data_unique.columns if data_unique[feature].dtype == 'object']
                data_encoded = data_unique.copy()
                for feature in categorical_features:
                    labels_ordered = data_unique.groupby([feature]).size().sort_values().index
                    labels_ordered = {k: i for i, k in enumerate(labels_ordered, 0)}
                    data_encoded[feature] = data_encoded[feature].map(labels_ordered)
                data = data_encoded  # Update the original dataset with encoded features
                st.write("Categorical features encoded successfully.")

                st.write("Performing feature scaling...")
                numeric_columns = data.select_dtypes(include=["int", "float"]).columns

                if len(numeric_columns) == 0:
                    st.write("No numeric columns found.")
                else:
                    scaler = MinMaxScaler()
                    data_scaled = data.copy()
                    data_scaled[numeric_columns] = scaler.fit_transform(data_scaled[numeric_columns])
                    data = data_scaled  # Update the original dataset with scaled features
                    st.write("Feature scaling performed successfully.")

                st.write("Downloading the dataset...")

                # Save the modified dataset to a file
                modified_dataset_filename = "modified_dataset.csv"
                # data.to_csv(modified_dataset_filename, index=False)
                st.write(data.head())
                st.write(data.shape)
















         # ending of basic information side bar


        #  starting of anomaly detection algorithms over here
        st.sidebar.header("Statistical Methods")
        anomaly_options = ["None",
                        "Z-Score/Standard Deviation",
                        "Boxplot",
                        "Probability Density Function",
                        "RSF",
                        "Benford law 1st digit", "Benford law 2nd digit", "Benford law 3rd digit"
        ]
        selected_anomalyAlgorithm = st.sidebar.selectbox("Choose statistical methods", anomaly_options)

        if selected_anomalyAlgorithm == "None":
            st.write(" ")
        elif selected_anomalyAlgorithm == "Z-Score/Standard Deviation":
            # Calculate the Z-score for each feature in the dataset
            z_scores = stats.zscore(data)

            # Set a threshold for anomaly detection
            threshold = 3

            # Identify outliers based on the Z-score exceeding the threshold
            outlier_indices = np.where(np.abs(z_scores) > threshold)[0]

            # Create a copy of the data with an "Anomaly" column indicating outliers
            data_with_anomalies_zscore = data.copy()
            data_with_anomalies_zscore['Anomaly'] = 0
            data_with_anomalies_zscore.iloc[outlier_indices, -1] = 1

            st.subheader("Data with Anomalies (Z-Score)")
            st.write(data_with_anomalies_zscore)

            selected_x_col = st.selectbox("Select X-axis column", data.columns)
            selected_y_col = st.selectbox("Select Y-axis column", data.columns)

            # Plot the results using Plotly Express
            fig = px.scatter(data_with_anomalies_zscore, x=selected_x_col, y=selected_y_col, color='Anomaly')
            fig.update_layout(title='Z-Score Anomaly Detection')

            # Save the Plotly figure as an HTML file
            fig_html_path = "plot.html"
            fig.write_html(fig_html_path)

            # Provide a button to open the Plotly chart in a new tab
            if st.button("Open Plotly Chart"):
                new_tab = webbrowser.get()
                new_tab.open(fig_html_path, new=2)

            # Counting the number of outliers
            num_outliers = len(outlier_indices)

            # Total number of data points
            total_data_points = len(data_with_anomalies_zscore)

            # Calculating the percentage of outliers
            percentage_outliers = (num_outliers / total_data_points) * 100

            st.write(f"Number of outliers: {num_outliers}")
            st.write(f"Percentage of outliers: {percentage_outliers:.2f}%")

            # Download the data with anomaly indicator
            st.write("Download the data with anomaly indicator (Z-Score)")
            st.download_button(
                label="Download",
                data=data_with_anomalies_zscore.to_csv(index=False),
                file_name="ZScoreAnomaly.csv",
                mime="text/csv"
            )

        elif selected_anomalyAlgorithm == "Boxplot":
            # Select the feature to visualize
            selected_feature = st.selectbox("Select a feature:", data.columns)

            # Generate the boxplot using Plotly Express
            fig = px.box(data, y=selected_feature)
            fig.update_layout(title="Boxplot of " + selected_feature)

            # Save the Plotly figure as an HTML file
            fig_html_path = "boxplot.html"
            fig.write_html(fig_html_path)

            # Provide a link to open the Plotly chart in a new tab
            if st.button("Open Boxplot"):
                new_tab = webbrowser.get()
                new_tab.open(fig_html_path, new=2)

            # Calculate interquartile range (IQR)
            Q1 = data[selected_feature].quantile(0.25)
            Q3 = data[selected_feature].quantile(0.75)
            IQR = Q3 - Q1

            # Calculate upper and lower limits
            lower_limit = Q1 - 1.5 * IQR
            upper_limit = Q3 + 1.5 * IQR

            # Find outliers and create the anomaly feature
            data['anomaly'] = 0  # Initialize anomaly feature as 0
            data.loc[(data[selected_feature] < lower_limit) | (data[selected_feature] > upper_limit), 'anomaly'] = 1

            # Calculate the percentage of outliers
            total_data_points = data.shape[0]
            total_outliers = data['anomaly'].sum()
            percentage_outliers = (total_outliers / total_data_points) * 100

            # Show the updated dataframe with the anomaly feature and the percentage of outliers
            st.write("Updated Dataframe:")
            st.write(data)

            # Download the dataframe with the feature name in the file name
            file_name = "Anomaly_" + selected_feature.replace(" ", "_") + ".csv"
            st.download_button(
                label="Download",
                data=data.to_csv(index=False),
                file_name=file_name,
                mime="text/csv"
            )

            st.write("Percentage of outliers: {:.2f}%".format(percentage_outliers))

        elif selected_anomalyAlgorithm == "Probability Density Function":
            # Select the feature to analyze
            selected_feature = st.selectbox("Select a feature:", data.columns)

            # Calculate mean and standard deviation
            mean = data[selected_feature].mean()
            std = data[selected_feature].std()

            # Create a range of values for the PDF
            x = np.linspace(data[selected_feature].min(), data[selected_feature].max(), 100)

            # Calculate the PDF using Gaussian distribution
            pdf = norm.pdf(x, mean, std)

            # Create a DataFrame for the PDF data
            pdf_data = pd.DataFrame({'x': x, 'pdf': pdf})

            # Plot the PDF using Plotly Express
            fig = px.line(pdf_data, x='x', y='pdf')
            fig.update_layout(
                title="Probability Density Function of " + selected_feature,
                xaxis_title=selected_feature,
                yaxis_title="PDF"
            )

            # Save the Plotly figure as an HTML file
            fig_html_path = "pdf_plot.html"
            fig.write_html(fig_html_path)

            # Provide a link to open the Plotly chart in a new tab
            if st.button("Open PDF Plot"):
                new_tab = webbrowser.get()
                new_tab.open(fig_html_path, new=2)

            # Apply the PDF to identify outliers
            threshold = 0.01  # Adjust the threshold as needed
            data['anomaly'] = 0  # Initialize anomaly feature as 0
            data.loc[data[selected_feature] < norm.ppf(threshold, mean, std), 'anomaly'] = 1

            # Calculate the percentage of outliers
            total_data_points = data.shape[0]
            total_outliers = data['anomaly'].sum()
            percentage_outliers = (total_outliers / total_data_points) * 100

            # Show the updated dataframe with the anomaly feature and the percentage of outliers
            st.write("Updated Dataframe:")
            st.write(data)

            file_name = "Anomaly_" + selected_feature.replace(" ", "_") + ".csv"
            st.download_button(
                label="Download",
                data=data.to_csv(index=False),
                file_name=file_name,
                mime="text/csv"
            )

            st.write("Percentage of outliers: {:.2f}%".format(percentage_outliers))


        elif selected_anomalyAlgorithm == "RSF":
            selected_feature = st.selectbox("Select a feature:", data.columns)

            # Calculate the relative strength factor (RSF)
            rsf = data[selected_feature] / data[selected_feature].rolling(window=14).mean()

            # Create a DataFrame for RSF data
            rsf_data = pd.DataFrame({'Data Point': data.index, 'RSF': rsf})

            # Plot the RSF using Plotly Express
            fig = px.line(rsf_data, x='Data Point', y='RSF')
            fig.update_layout(
                title="Relative Strength Factor of " + selected_feature,
                xaxis_title="Data Point",
                yaxis_title="RSF"
            )

            # Save the Plotly figure as an HTML file
            fig_html_path = "rsf_plot.html"
            fig.write_html(fig_html_path)

            # Provide a link to open the Plotly chart in a new tab
            if st.button("Open RSF Plot"):
                new_tab = webbrowser.get()
                new_tab.open(fig_html_path, new=2)

            # Apply a threshold to identify outliers
            threshold = 1.5  # Adjust the threshold as needed
            data['anomaly'] = np.where(rsf > threshold, 1, 0)

            # Calculate the percentage of outliers
            total_data_points = data.shape[0]
            total_outliers = data['anomaly'].sum()
            percentage_outliers = (total_outliers / total_data_points) * 100

            # Show the updated dataframe with the anomaly feature and the percentage of outliers
            st.write("Updated Dataframe:")
            st.write(data)

            file_name = "Anomaly_" + selected_feature.replace(" ", "_") + ".csv"
            st.download_button(
                label="Download",
                data=data.to_csv(index=False),
                file_name=file_name,
                mime="text/csv"
            )

            st.write("Percentage of outliers: {:.2f}%".format(percentage_outliers))

        elif selected_anomalyAlgorithm == "Benford law 1st digit":
            # Assuming you have a DataFrame named 'data' with a column of numeric data named 'selected_feature'
            selected_feature = st.selectbox("Select a feature:", data.columns)

            # Calculate the distribution of first digits using Benford's Law
            counts, benford = calculate_first_digit(data[selected_feature])

            import plotly.graph_objs as go
            import plotly.io as pio
            # import pandas as pd
            # import numpy as np

            # Calculate the distribution of first digits using Benford's Law
            counts, benford = calculate_first_digit(data[selected_feature])

            # Exclude the first digit (0) from the observed distribution
            #counts = counts.iloc[1:]

            # Create the observed and expected bar plots
            observed_trace = go.Bar(x=counts.index, y=counts * 100, name='Observed')
            expected_trace = go.Scatter(x=np.arange(0, 10), y=benford * 100, mode='lines', name='Expected')

            # Create the layout
            layout = go.Layout(
                title="Benford's Law Analysis of " + selected_feature,
                xaxis=dict(title="First Digit"),
                yaxis=dict(title="Percentage"),
                legend=dict(x=0, y=1)
            )

            # Create the figure and add the traces
            fig = go.Figure(data=[observed_trace, expected_trace], layout=layout)

            # Display the figure
            fig.show()

            # Option to save the plot
            save_plot_option = st.checkbox("Save plot")
            if save_plot_option:
                file_name = "Benford_Plot_" + selected_feature.replace(" ", "_") + ".html"
                pio.write_html(fig, file_name)
                st.write(f"Plot saved as {file_name}")

            # Calculate the deviation from expected percentages
            deviation = (counts - benford) * 100

            # Create the results DataFrame
            results = pd.DataFrame(
                {'Digit': counts.index, 'Observed (%)': counts * 100, 'Expected (%)': benford * 100,
                 'Deviation (%)': deviation})

            # Option to save the results as CSV
            save_csv_option = st.checkbox("Save results as CSV")
            if save_csv_option:
                file_name = "Benford_Analysis_" + selected_feature.replace(" ", "_") + ".csv"
                st.download_button(
                    label="Download CSV",
                    data=results.to_csv(index=False),
                    file_name=file_name,
                    mime="text/csv"
                )

            # Display the results DataFrame
            st.write("Results:")
            st.write(results)

        #  ending of anomaly detection algorithms over here

        elif selected_anomalyAlgorithm == "Benford law 2nd digit":
            # Assuming you have a DataFrame named 'data' with a column of numeric data named 'selected_feature'

            selected_feature = st.selectbox("Select a feature:", data.columns)

            # Calculate the distribution of second digits using Benford's Law
            counts, benford = calculate_2th_digit(data[selected_feature])

            import plotly.graph_objs as go
            import plotly.io as pio

            # Exclude the first digit (0) from the observed distribution
            #counts = counts.iloc[1:]

            # Create the observed and expected bar plots
            observed_trace = go.Bar(x=counts.index, y=counts * 100, name='Observed', marker=dict(color='blue'))
            expected_trace = go.Scatter(x=np.arange(0, 100), y=benford * 100, mode='lines', line=dict(color='red'),
                                        name='Expected')

            # Create the layout
            layout = go.Layout(
                title="Benford's Law Analysis of Second Digit in " + selected_feature,
                xaxis=dict(title="Second Digit"),
                yaxis=dict(title="Percentage"),
                legend=dict(x=0, y=1)
            )

            # Create the figure and add the traces
            fig = go.Figure(data=[observed_trace, expected_trace], layout=layout)

            # Display the figure
            fig.show()

            # Option to save the plot
            save_plot_option = st.checkbox("Save plot")
            if save_plot_option:
                file_name = "Benford_Plot_2nd_Digit_" + selected_feature.replace(" ", "_") + ".html"
                pio.write_html(fig, file_name)
                st.write(f"Plot saved as {file_name}")

            # Calculate the deviation from expected percentages
            deviation = (counts - benford) * 100

            # Create the results DataFrame
            results = pd.DataFrame(
                {'Digit': counts.index, 'Observed (%)': counts * 100, 'Expected (%)': benford[:len(counts)] * 100,
                 'Deviation (%)': deviation})

            # Option to save the results as CSV
            save_csv_option = st.checkbox("Save results as CSV")
            if save_csv_option:
                file_name = "Benford_Analysis_2nd_Digit_" + selected_feature.replace(" ", "_") + ".csv"
                st.download_button(
                    label="Download CSV",
                    data=results.to_csv(index=False),
                    file_name=file_name,
                    mime="text/csv"
                )

            # Display the results DataFrame
            st.write("Results:")
            st.write(results)


        elif selected_anomalyAlgorithm == "Benford law 3rd digit":
            # Assuming you have a DataFrame named 'data' with a column of numeric data named 'selected_feature'

            selected_feature = st.selectbox("Select a feature:", data.columns)

            # Calculate the distribution of second digits using Benford's Law
            counts, benford = calculate_3th_digit(data[selected_feature])

            import plotly.graph_objs as go
            import plotly.io as pio

            # Exclude the first digit (0) from the observed distribution
            #counts = counts.iloc[1:]

            # Create the observed and expected bar plots
            observed_trace = go.Bar(x=counts.index, y=counts * 100, name='Observed', marker=dict(color='blue'))
            expected_trace = go.Scatter(x=np.arange(0, 1000), y=benford * 100, mode='lines', line=dict(color='red'),
                                        name='Expected')

            # Create the layout
            layout = go.Layout(
                title="Benford's Law Analysis of Second Digit in " + selected_feature,
                xaxis=dict(title="Second Digit"),
                yaxis=dict(title="Percentage"),
                legend=dict(x=0, y=1)
            )

            # Create the figure and add the traces
            fig = go.Figure(data=[observed_trace, expected_trace], layout=layout)

            # Display the figure
            fig.show()

            # Option to save the plot
            save_plot_option = st.checkbox("Save plot")
            if save_plot_option:
                file_name = "Benford_Plot_3nd_Digit_" + selected_feature.replace(" ", "_") + ".html"
                pio.write_html(fig, file_name)
                st.write(f"Plot saved as {file_name}")

            # Calculate the deviation from expected percentages
            deviation = (counts - benford) * 100

            # Create the results DataFrame
            results = pd.DataFrame(
                {'Digit': counts.index, 'Observed (%)': counts * 100, 'Expected (%)': benford[:len(counts)] * 100,
                 'Deviation (%)': deviation})

            # Option to save the results as CSV
            save_csv_option = st.checkbox("Save results as CSV")
            if save_csv_option:
                file_name = "Benford_Analysis_3nd_Digit_" + selected_feature.replace(" ", "_") + ".csv"
                st.download_button(
                    label="Download CSV",
                    data=results.to_csv(index=False),
                    file_name=file_name,
                    mime="text/csv"
                )

            # Display the results DataFrame
            st.write("Results:")
            st.write(results)
        #  starting of anomaly detection algorithms over here
        st.sidebar.header("Machine Learning Methods")
        anomaly_options = ["None",
                        "Isolation Forest",
                        "Gaussian Mixture Models (GMM)",
                        "Kernel Density Estimation (KDE)",
                        "K-Means",
                        "DBSCAN"
        ]
        selected_anomalyAlgorithm = st.sidebar.selectbox("Choose density-based method", anomaly_options)

        if selected_anomalyAlgorithm == "None":
            st.write(" ")


        elif selected_anomalyAlgorithm == "Isolation Forest":
            # Applying the anomaly detection
            data_with_anomalies_IsolationForest = apply_anomaly_detection_IsolationForest(data)

            st.subheader("Data with Anomalies")
            st.write(data_with_anomalies_IsolationForest)

            selected_x_col = st.selectbox("Select X-axis column", data.columns)
            selected_y_col = st.selectbox("Select Y-axis column", data.columns)

            # Plot the results using Plotly Express
            fig = px.scatter(data_with_anomalies_IsolationForest, x=selected_x_col, y=selected_y_col, color='Anomaly')
            fig.update_layout(title='Isolation Forest Anomaly Detection')

            # Save the Plotly figure as an HTML file
            fig_html_path = "isolation_forest_plot.html"
            fig.write_html(fig_html_path)

            # Provide a link to open the Plotly chart in a new tab
            if st.button("Open Isolation Forest Plot"):
                new_tab = webbrowser.get()
                new_tab.open(fig_html_path, new=2)

            st.write("Download the data with anomaly indicator")
            st.download_button(
                label="Download",
                data=data_with_anomalies_IsolationForest.to_csv(index=False),
                file_name="IsolationForestAnomaly.csv",
                mime="text/csv"
            )

            # Count the number of anomalies
            num_anomalies = data_with_anomalies_IsolationForest['Anomaly'].sum()

            # Total number of data points
            total_data_points = len(data_with_anomalies_IsolationForest)

            # Calculate the percentage of anomalies
            percentage_anomalies = (num_anomalies / total_data_points) * 100

            st.write(f"Number of anomalies: {num_anomalies}")
            st.write(f"Percentage of anomalies: {percentage_anomalies:.2f}%")

        elif selected_anomalyAlgorithm == "Kernel Density Estimation (KDE)":
            # Perform anomaly detection using Kernel Density Estimation
            kde = KernelDensity()
            kde.fit(data)
            log_densities = kde.score_samples(data)
            threshold = np.percentile(log_densities, 5)  # Adjust the percentile as needed

            # Identify outliers based on log densities below the threshold
            outlier_indices = np.where(log_densities < threshold)[0]

            # Create a copy of the data with an "Anomaly" column indicating outliers
            data_with_anomalies_kde = data.copy()
            data_with_anomalies_kde['Anomaly'] = 0
            data_with_anomalies_kde.iloc[outlier_indices, -1] = 1

            st.subheader("Data with Anomalies (KDE)")
            st.write(data_with_anomalies_kde)

            selected_feature = st.selectbox("Select a feature", data.columns)

            # Create a DataFrame for the KDE results
            kde_data = pd.DataFrame({'Feature': data[selected_feature], 'Density': np.exp(log_densities)})

            # Plot the results using Plotly Express
            fig = px.line(kde_data, x='Feature', y='Density')
            fig.update_layout(
                title='Kernel Density Estimation Anomaly Detection',
                xaxis_title=selected_feature,
                yaxis_title='Density'
            )

            # Save the Plotly figure as an HTML file
            fig_html_path = "kde_plot.html"
            fig.write_html(fig_html_path)

            # Provide a link to open the Plotly chart in a new tab
            if st.button("Open KDE Plot"):
                new_tab = webbrowser.get()
                new_tab.open(fig_html_path, new=2)

            # Counting the number of anomalies
            num_anomalies = data_with_anomalies_kde['Anomaly'].sum()

            # Total number of data points
            total_data_points = len(data_with_anomalies_kde)

            # Calculating the percentage of anomalies
            percentage_anomalies = (num_anomalies / total_data_points) * 100

            st.write(f"Number of anomalies: {num_anomalies}")
            st.write(f"Percentage of anomalies: {percentage_anomalies:.2f}%")

            # Download the data with anomaly indicator
            st.write("Download the data with anomaly indicator (KDE)")
            st.download_button(
                label="Download",
                data=data_with_anomalies_kde.to_csv(index=False),
                file_name="KDEAnomaly.csv",
                mime="text/csv"
            )









        elif selected_anomalyAlgorithm == "K-Means":
            # Applying K-means clustering
            kmeans = KMeans(n_clusters=2)  # You can adjust the number of clusters as needed
            kmeans.fit(data)

            # Predicting cluster labels
            cluster_labels = kmeans.predict(data)
            data_with_clusters = data.copy()
            data_with_clusters['Cluster'] = cluster_labels

            # Calculate the percentage of outliers
            outlier_percentage = (data_with_clusters['Cluster'].value_counts()[1] / len(data_with_clusters)) * 100

            # Create an anomaly indicator
            data_with_clusters['Anomaly'] = np.where(data_with_clusters['Cluster'] == 1, 1, 0)

            st.subheader("Data with Anomaly Indicator")
            st.write(data_with_clusters)

            selected_x_col = st.selectbox("Select X-axis column", data.columns)
            selected_y_col = st.selectbox("Select Y-axis column", data.columns)

            # Plot the results using Plotly Express
            fig = px.scatter(data_with_clusters, x=selected_x_col, y=selected_y_col, color='Cluster')
            fig.update_layout(title='K-means Clustering')

            # Save the Plotly figure as an HTML file
            fig_html_path = "kmeans_plot.html"
            fig.write_html(fig_html_path)

            # Provide a link to open the Plotly chart in a new tab
            if st.button("Open K-Means Plot"):
                new_tab = webbrowser.get()
                new_tab.open(fig_html_path, new=2)

            # Counting the number of anomalies
            num_anomalies = data_with_clusters['Anomaly'].sum()

            # Total number of data points
            total_data_points = len(data_with_clusters)

            # Calculating the percentage of anomalies
            percentage_anomalies = (num_anomalies / total_data_points) * 100

            st.write(f"Number of anomalies: {num_anomalies}")
            st.write(f"Percentage of anomalies: {percentage_anomalies:.2f}%")

            # Download the data with anomaly indicator
            st.write("Download the data with anomaly indicator")
            st.download_button(
                label="Download",
                data=data_with_clusters.to_csv(index=False),
                file_name="KMeansAnomaly.csv",
                mime="text/csv"
            )




        #  ending of anomaly detection algorithms over here


















        # starting of visualization side bar over here



        st.sidebar.header("Deep Learning Methods")
        anomaly_options = ["None", "Autoencoder"]
        selected_anomalyAlgorithm = st.sidebar.selectbox("Choose an algorithm", anomaly_options)

        if selected_anomalyAlgorithm == "None":
            st.write(" ")

        elif selected_anomalyAlgorithm == "Autoencoder":
            # Define the autoencoder model
            input_dim = data.shape[1]  # Assuming data is a DataFrame with the appropriate shape
            encoding_dim = 64  # Adjust the encoding dimension as needed
            input_layer = Input(shape=(input_dim,))
            encoded = Dense(encoding_dim, activation='relu')(input_layer)
            decoded = Dense(input_dim, activation='sigmoid')(encoded)
            autoencoder = Model(inputs=input_layer, outputs=decoded)
            autoencoder.compile(optimizer='adam', loss='mse')

            # Train the autoencoder
            autoencoder.fit(data, data, epochs=10, batch_size=32)  # Adjust epochs and batch size as needed

            # Obtain the reconstructed data
            reconstructed_data = autoencoder.predict(data)

            # Create a DataFrame with the data and the anomaly indicator
            data_with_anomalies_autoencoder = data.copy()
            data_with_anomalies_autoencoder['Anomaly'] = tf.keras.losses.mean_squared_error(data, reconstructed_data).numpy() > 95
            data_with_anomalies_autoencoder['Anomaly'] = data_with_anomalies_autoencoder['Anomaly'].astype(int)

            st.subheader("Data with Anomalies")
            st.write(data_with_anomalies_autoencoder)

            selected_x_col = st.selectbox("Select X-axis column", data.columns)
            selected_y_col = st.selectbox("Select Y-axis column", data.columns)

            # Plot the results using Plotly Express
            fig = px.scatter(data_with_anomalies_autoencoder, x=selected_x_col, y=selected_y_col, color='Anomaly')
            fig.update_layout(title='Autoencoder Anomaly Detection')

            # Save the Plotly figure as an HTML file
            fig_html_path = "autoencoder_plot.html"
            fig.write_html(fig_html_path)

            # Provide a link to open the Plotly chart in a new tab
            if st.button("Open Autoencoder Plot"):
                new_tab = webbrowser.get()
                new_tab.open(fig_html_path, new=2)

            st.write("Download the data with anomaly indicator")
            st.download_button(
                label="Download",
                data=data_with_anomalies_autoencoder.to_csv(index=False),
                file_name="AutoencoderAnomaly.csv",
                mime="text/csv"
            )

            # Count the number of anomalies
            num_anomalies = data_with_anomalies_autoencoder['Anomaly'].sum()

            # Total number of data points
            total_data_points = len(data_with_anomalies_autoencoder)

            # Calculate the percentage of anomalies
            percentage_anomalies = (num_anomalies / total_data_points) * 100

            st.write(f"Number of anomalies: {num_anomalies}")
            st.write(f"Percentage of anomalies: {percentage_anomalies:.2f}%")



    
if __name__ == "__main__":
    main()


