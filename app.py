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

import base64


import streamlit as st

from matplotlib import style
style.use("ggplot")
st.set_option('deprecation.showPyplotGlobalUse', False)
# Define the HTML code with CSS for the marquee
marquee_html = """

<style>
.marquee {
    width: 100%;
    overflow: hidden;
    white-space: nowrap;
}

.marquee span {
    display: inline-block;
    padding-left: 100%;
    animation: marquee 10s linear infinite;
}
.logo {
    width: 150px;
    position: relative;
    bottom: 20px;
}

#right {
    text-align: right;
}

h1 {
    font-size: 100px;
    text-align: left;
    position: relative;
    top: 100px;
}

@keyframes marquee {
    0% { transform: translate(0, 0); }
    100% { transform: translate(-100%, 0); }
}
</style>


<h1 class="heading">Infrared</h1>
<div id="right">
    <img src="https://revoquant.com/assets/img/logo/logo-dark.png" class="logo">
</div>
<div class="marquee">
    <span>This application offers a sophisticated approach to comprehending your dataset and identifying outliers.</span>
</div>
<center>
<br>
<br>
</center>
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






def main():
    

    st.header("Upload you csv data")
    data_file = st.file_uploader("Upload CSV", type=["csv"])

    if data_file is not None:
        data = pd.read_csv(data_file)

    

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

            # code for taking the input from two csv files over here
            st.header("Upload you first csv file")
            data_file1 = st.file_uploader("Upload CSV", type=["csv"],key=2)

            if data_file1 is not None:
                data1 = pd.read_csv(data_file1)
                st.write(data1.head(2))

            st.header("Upload you second csv file")
            data_file2 = st.file_uploader("Upload CSV", type=["csv"],key=3)

            if data_file2 is not None:
                data2 = pd.read_csv(data_file2)
                st.write(data2.head(2))



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
                        "Probabily Density function.",
                        "RSF",
                        "Benford law"
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

            # Plot the results
            st.subheader("Anomaly Detection Plot (Z-Score)")
            fig, ax = plt.subplots()
            ax.scatter(data_with_anomalies_zscore[selected_x_col], data_with_anomalies_zscore[selected_y_col],
                    c=data_with_anomalies_zscore['Anomaly'], cmap='RdYlBu')
            ax.set_xlabel(selected_x_col)
            ax.set_ylabel(selected_y_col)
            ax.set_title('Z-Score Anomaly Detection')
            st.pyplot(fig)

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


    
    


        #  ending of anomaly detection algorithms over here


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

            # Plot the results
            st.subheader("Anomaly Detection Plot")
            fig, ax = plt.subplots()
            ax.scatter(data_with_anomalies_IsolationForest[selected_x_col], data_with_anomalies_IsolationForest[selected_y_col],
                    c=data_with_anomalies_IsolationForest['Anomaly'], cmap='RdYlBu')
            ax.set_xlabel(selected_x_col)
            ax.set_ylabel(selected_y_col)
            ax.set_title('Isolation Forest Anomaly Detection')
            st.pyplot(fig)

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

            # Plot the results
            st.subheader("Anomaly Detection Plot (KDE)")
            fig, ax = plt.subplots()
            ax.plot(data[selected_feature], np.exp(log_densities), 'k.', markersize=2)
            ax.fill_between(data[selected_feature], np.exp(log_densities), alpha=0.5)
            ax.set_xlabel(selected_feature)
            ax.set_ylabel("Density")
            ax.set_title('Kernel Density Estimation Anomaly Detection')
            st.pyplot(fig)

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

            # Plot the results
            st.subheader("K-means Clustering Plot")
            fig, ax = plt.subplots()
            ax.scatter(data_with_clusters[selected_x_col], data_with_clusters[selected_y_col], c=data_with_clusters['Cluster'], cmap='viridis')
            ax.set_xlabel(selected_x_col)
            ax.set_ylabel(selected_y_col)
            ax.set_title('K-means Clustering')
            st.pyplot(fig)



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

            # Plot the results
            st.subheader("Anomaly Detection Plot")
            fig, ax = plt.subplots()
            ax.scatter(data_with_anomalies_autoencoder[selected_x_col], data_with_anomalies_autoencoder[selected_y_col],
                    c=data_with_anomalies_autoencoder['Anomaly'], cmap='RdYlBu')
            ax.set_xlabel(selected_x_col)
            ax.set_ylabel(selected_y_col)
            ax.set_title('Autoencoder Anomaly Detection')
            st.pyplot(fig)

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


