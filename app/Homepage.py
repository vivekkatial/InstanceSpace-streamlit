import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import os
import pandas as pd
import numpy as np
import zipfile

st.set_page_config(
    page_title="Instance Space Analysis",
    page_icon="📊",
)

# delete the temp_extracted_files folder if it exists on first run
if os.path.exists("temp_extracted_files"):
    os.system("rm -r temp_extracted_files")


def st_authenticator():
    with open('.secrets/config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)
        file.close()

    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['preauthorized']
    )

    return authenticator

authenticator  = st_authenticator()
name, authentication_status, username = authenticator.login('main', fields = {'Form name': 'Login'})

if authentication_status:
    st.session_state.authentication_status = True
    authenticator.logout('**Logout**', 'main', key='unique_key')
elif authentication_status is False:
    st.session_state.authentication_status = False
    st.error('Username/password is incorrect')
elif authentication_status is None:
    st.session_state.authentication_status = None
    st.warning('Please enter your username and password')

if st.session_state.authentication_status:


    st.write("# :wave: Instance Space Analysis App :wave:")
    st.markdown(
        """
        This StreamLit app showcases the Instance Space Analysis (ISA) toolkit, designed to assess algorithmic performance and power across diverse problem instances. The ISA methodology models the relationship between structural properties of instances and algorithm performance, offering insights through visualizations and metrics.

        - To use the app -- first locally run the ISA toolkit on your data. You can find the toolkit on [Github here.](https://github.com/andremun/InstanceSpace)
        - Once you have the ISA toolkit output, you can upload the data here and explore the results. The app will provide visualizations and metrics to help you understand the relationship between your instances and algorithm performance.
        - For more information on the ISA methodology, please refer to the [MATILDA website](https://matilda.unimelb.edu.au/matilda/).
        """
    )

    # Upload zip file
    uploaded_file = st.file_uploader("Upload your ISA toolkit output zip file", type="zip")

    def process_zip_file(uploaded_file):
        if uploaded_file is not None:
            # Extract files from zip file
            with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                zip_ref.extractall("temp_extracted_files")

            data_dir = "temp_extracted_files"
            experiments = os.listdir(data_dir)
            
            # Remove .DS_Store file if it exists
            if ".DS_Store" in experiments:
                experiments.remove(".DS_Store")

            experiment = st.sidebar.selectbox("Select ISA Experiment", experiments)
            experiment_dir = os.path.join(data_dir, experiment)

            # Define the required files
            required_files = [
                "coordinates.csv",
                "metadata.csv",
                "svm_table.csv",
                "bounds_prunned.csv",
                "feature_process.csv",
                "feature_raw.csv",
                "algorithm_raw.csv",
                "algorithm_process.csv",
                "algorithm_svm.csv",
                "portfolio_svm.csv"
            ]
            
            # Check if all required files exist
            for file in required_files:
                if not os.path.exists(os.path.join(experiment_dir, file)):
                    st.error(f"Missing required file: {file}")
                    return

            # Read in the coordinates.csv file for the experiment 
            d_coords = pd.read_csv(os.path.join(experiment_dir, "coordinates.csv"))
            # Read in the metadata file for the experiment
            d_metadata = pd.read_csv(os.path.join(experiment_dir, "metadata.csv"))
            # select only the columns that are needed
            d_metadata = d_metadata[["Instances", "Source"]]
            # Merge the metadata with the coordinates ("Row" = "Instances")
            d_coords = pd.merge(d_coords, d_metadata, left_on="Row", right_on="Instances")
            # Drop the "Instances" column
            d_coords.drop(columns="Instances", inplace=True)
            # If the source contains Evolution keep the str else fill with Original
            d_coords['Evolution'] = np.where(d_coords['Source'].str.contains('Evolution'), d_coords['Source'], 'Original')
            
            # Read the SVM table
            d_svm = pd.read_csv(os.path.join(experiment_dir, "svm_table.csv"))
            
            # Read the bounds file
            d_bounds = pd.read_csv(os.path.join(experiment_dir, "bounds_prunned.csv"))
            if experiment in ["qaoa-param-inform-pub", "qaoa-classical-opts-init"]:
                d_bounds = pd.read_csv(os.path.join(experiment_dir, "bounds.csv"))
            
            # Read the feature table
            d_features = pd.read_csv(os.path.join(experiment_dir, "feature_process.csv"))
            d_features_raw = pd.read_csv(os.path.join(experiment_dir, "feature_raw.csv"))
            # Read the algorithm table
            d_algorithm_raw = pd.read_csv(os.path.join(experiment_dir, "algorithm_raw.csv"))
            d_algorithm_process = pd.read_csv(os.path.join(experiment_dir, "algorithm_process.csv"))
            d_algorithm_binary = pd.read_csv(os.path.join(experiment_dir, "algorithm_bin.csv"))

            
            # Read the SVM table and best algorithm
            d_svm_preds = pd.read_csv(os.path.join(experiment_dir, "algorithm_svm.csv"))
            d_svm_selection = pd.read_csv(os.path.join(experiment_dir, "portfolio_svm.csv"))
            # Find the best algorithm for each row
            
            # Create a new df for best_algorithm from d_algorithm_raw
            # The column with the minimum value is the best algorithm (excluding the first column)
            d_best_algo = pd.DataFrame()
            d_best_algo['Best_Algorithm'] = d_algorithm_raw.iloc[:, 1:].idxmin(axis=1)
            d_best_algo['Row'] = d_algorithm_raw['Row']
            # Reshuffle the columns
            d_best_algo = d_best_algo[['Row', 'Best_Algorithm']]

            st.success("Files processed successfully!")

            # Store experiment name in the session state
            st.session_state.experiment = experiment
            # Store dataframes in session state
            st.session_state.d_coords = d_coords
            st.session_state.d_svm = d_svm
            st.session_state.d_bounds = d_bounds
            st.session_state.d_features = d_features
            st.session_state.d_features_raw = d_features_raw
            st.session_state.d_algorithm_raw = d_algorithm_raw
            st.session_state.d_algorithm_process = d_algorithm_process
            st.session_state.d_algorithm_binary = d_algorithm_binary
            st.session_state.d_svm_preds = d_svm_preds
            st.session_state.d_svm_selection = d_svm_selection
            st.session_state.d_best_algo = d_best_algo

            # Read in all the algorithms (excluding the first column from d_best_algo)
            algorithms = d_algorithm_raw.columns[1:]
            # Check if footprint files exist for each algorithm
            for algo in algorithms:
                if not os.path.exists(os.path.join(experiment_dir, f"footprint_{algo}_best.csv")):
                    st.warning(f"Missing 'best' footprint file for algorithm: {algo}")
                if not os.path.exists(os.path.join(experiment_dir, f"footprint_{algo}_good.csv")):
                    st.warning(f"Missing 'good' footprint file for algorithm: {algo}")
            


    if uploaded_file:
        process_zip_file(uploaded_file)

    # Display the data if it exists in session state
    if "d_coords" in st.session_state:
        st.write("Coordinates DataFrame", st.session_state.d_coords)
    if "d_svm" in st.session_state:
        st.write("SVM Table DataFrame", st.session_state.d_svm)
    if "d_bounds" in st.session_state:
        st.write("Bounds DataFrame", st.session_state.d_bounds)
    if "d_features" in st.session_state:
        st.write("Feature Process DataFrame", st.session_state.d_features)
    if "d_features_raw" in st.session_state:
        st.write("Feature Raw DataFrame", st.session_state.d_features_raw)
    if "d_algorithm_raw" in st.session_state:
        st.write("Algorithm Raw DataFrame", st.session_state.d_algorithm_raw)
    if "d_algorithm_process" in st.session_state:
        st.write("Algorithm Process DataFrame", st.session_state.d_algorithm_process)
    if "d_algorithm_binary" in st.session_state:
        st.write("Algorithm Binary DataFrame", st.session_state.d_algorithm_binary)
    if "d_svm_preds" in st.session_state:
        st.write("SVM Predictions DataFrame", st.session_state.d_svm_preds)
    if "d_svm_selection" in st.session_state:
        st.write("SVM Selection DataFrame", st.session_state.d_svm_selection)
    if "d_best_algo" in st.session_state:
        st.write("Best Algorithm DataFrame", st.session_state.d_best_algo)

    st.markdown(
        """
        **To continue to the analysis, click on the "Instance Space Analysis" page in the sidebar.**
        """
    )