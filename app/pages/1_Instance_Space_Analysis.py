import streamlit as st
import plotly.io as pio
import plotly.graph_objs as go
import plotly.express as px
import json
import numpy as np
import pandas as pd
import os
import shutil
import scipy.io
import sys

from src.visualize import *

st.set_page_config(page_title="Instance Space Analysis", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

st.markdown("# Instance Space Analysis")

# Check if the experiment and dataframes are in the session state
if "experiment" in st.session_state:
    experiment = st.session_state.experiment
    st.markdown(f"### Showing data for {experiment}")

    d_coords = st.session_state.d_coords
    d_svm = st.session_state.d_svm
    d_bounds = st.session_state.d_bounds
    d_features = st.session_state.d_features
    d_features_raw = st.session_state.d_features_raw
    d_algorithm_raw = st.session_state.d_algorithm_raw
    d_algorithm_process = st.session_state.d_algorithm_process
    d_svm_preds = st.session_state.d_svm_preds
    d_svm_selection = st.session_state.d_svm_selection
    d_best_algo = st.session_state.d_best_algo

    # Read the MAT file for algorithm labels
    mat = scipy.io.loadmat(os.path.join("temp_extracted_files", experiment, "model.mat"))
    algos = mat["data"]["algolabels"]
    algos = np.array([item for sublist in algos.flat for item in sublist.flat])
    algos = [item for sublist in algos for item in sublist]

    ### DISPLAY THE DATA ###
    # Display the SVM table
    st.subheader("SVM")

    d_svm_display = d_svm[["Row","Probability_of_good", "CV_model_accuracy", "CV_model_precision"]].style.format({
        "Probability_of_good": "{:.2%}",
        "CV_model_accuracy": "{:.2f}",
        "CV_model_precision": "{:.2f}"    
    })

    st.dataframe(d_svm_display)

    # Function to simulate progress for demonstration purposes
    def update_progress_bar(progress_bar, progress_text, iteration, total):
        progress_percentage = int((iteration / total) * 100)
        progress_bar.progress(progress_percentage)
        progress_text.text(f"Progress: {progress_percentage}%")

    # Download all the plots as a zip file
    download_all = st.button("Download All Plots")
    if download_all:
        st.write("Downloading all plots...")

        progress_bar = st.progress(0)
        progress_text = st.empty()

        # Create a temporary directory to store the plots
        os.makedirs("temp", exist_ok=True)

        # Make the source distribution plot
        source_fig = plot_source_distribution(d_coords, d_bounds)
        source_fig.update_layout(width=600, height=600)
        source_fig.update_layout(legend=dict(orientation="h", xanchor="center", x=0.5, y=-0.2))
        source_fig.write_image(os.path.join("temp", "source_distribution.png"), scale=3)
        
        # Update progress bar
        update_progress_bar(progress_bar, progress_text, 1, 14)

        # Create a vector for all features
        features = d_features.columns[1:]
        for i, feature in enumerate(features, start=2):
            fig_feat = plot_feature_distribution(d_features, d_coords, feature)
            fig_feat.update_layout(width=600, height=600)
            fig_feat.write_image(os.path.join("temp", f"feature_{feature}_distribution.png"), scale=3)

            update_progress_bar(progress_bar, progress_text, 2 + (i+1)/len(features), 14)
            
            # Do the raw features
            fig_feat_raw = plot_feature_distribution(d_features_raw, d_coords, feature)
            fig_feat_raw.update_layout(width=600, height=600)
            fig_feat_raw.write_image(os.path.join("temp", f"feature_{feature}_raw_distribution.png"), scale=3)

        # Update progress bar
        update_progress_bar(progress_bar, progress_text, 3, 14)

        # Create a vector for all algorithms
        algorithms = d_algorithm_raw.columns[1:]
        for i, algorithm in enumerate(algorithms, start=4):
            fig_algo = plot_performance_distribution(d_algorithm_process, d_coords, algorithm)
            fig_algo.update_layout(width=600, height=600)
            fig_algo.write_image(os.path.join("temp", f"performance_{algorithm}_distribution.png"), scale=3)

            # Do the raw features
            fig_algo_raw = plot_performance_distribution(d_algorithm_raw, d_coords, algorithm)
            fig_algo_raw.update_layout(width=600, height=600)
            fig_algo_raw.write_image(os.path.join("temp", f"performance_{algorithm}_raw_distribution.png"), scale=3)

        # Create the best algorithm plot
        fig_best_algo = plot_best_algorithm(d_coords, d_best_algo)
        fig_best_algo.update_layout(width=600, height=600)
        fig_best_algo.update_layout(legend=dict(orientation="h", xanchor="center", x=0.5, y=-0.2))
        fig_best_algo.write_image(os.path.join("temp", "best_algorithm_distribution.png"), scale=3)
        
        # Update progress bar
        update_progress_bar(progress_bar, progress_text, 12, 14)

        # Create the SVM selection plot
        for algorithm in algos:
            fig_svm = plot_svm_selection_single_algo(d_coords, d_svm_preds, algorithm, "temp_extracted_files/" + experiment, show_footprints=True)
            fig_svm.update_layout(width=600, height=600)
            fig_svm.write_image(os.path.join("temp", f"svm_selection_{algorithm}.png"), scale=3)
            update_progress_bar(progress_bar, progress_text, 13, 14)

        # Create the SVM selector plot
        fig_svm_selector = plot_svm_selector(d_coords, d_svm_preds, d_svm, experiment_dir="temp_extracted_files/" + experiment, show_footprints=True)
        fig_svm_selector.update_layout(width=600, height=600)
        fig_svm_selector.update_layout(legend=dict(orientation="h", xanchor="center", x=0.5, y=-0.2))
        fig_svm_selector.write_image(os.path.join("temp", "svm_selection.png"), scale=3)
        
        # Update progress bar
        update_progress_bar(progress_bar, progress_text, 14, 14)

        def callback():
            st.balloons()

        # Create a zip file with all the plots
        shutil.make_archive("temp", "zip", "temp")
        # Download the zip file

        with open("temp.zip", "rb") as f:            
            st.download_button(
                label="Download as Zip",
                data=f,
                file_name=f"{experiment}.zip",
                mime="application/zip",
                key="callback",
            )

        st.write("Download complete.")
        progress_bar.progress(100)
        progress_text.text("Progress: 100%")
        shutil.rmtree("temp")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Source Distribution")
        fig = plot_source_distribution(d_coords, d_bounds)
        st.plotly_chart(fig)
        download_source = st.button("Download Plot", key="source_plot")
        if download_source:
            st.write("Downloading plot...")
            fig.update_layout(width=600, height=600)
            fig.update_layout(legend=dict(orientation="h", xanchor="center", x=0.5, y=-0.2))
            fig.write_image(os.path.join("temp_extracted_files", experiment, "source_distribution.png"), scale=3)
            st.write("Download complete.")

    with col2:
        st.subheader("Transformation")
        dataframe = pd.read_csv(f"temp_extracted_files/{experiment}/projection_matrix.csv")
        dataframe = dataframe.T
        projection_matrix = dataframe.iloc[1:].round(2).astype(str)
        projection_matrix = projection_matrix.to_latex(index=False, escape=False, column_format='', header=False). \
            replace('\\begin{tabular}', '\\begin{bmatrix}').replace('\\end{tabular}', '\\end{bmatrix}'). \
            replace('\\toprule', '').replace('\\midrule', '').replace('\\bottomrule', ''). \
            replace('\\{c\\}', '')

        sel_feats = dataframe.reset_index().iloc[1:, 0]
        sel_feats = sel_feats.apply(lambda x: f'\\text{{{x}}}').str.replace('_', ' ')
        sel_feats = sel_feats.to_latex(index=False, escape=False, column_format='', header=False). \
            replace('\\begin{tabular}', '\\begin{bmatrix}').replace('\\end{tabular}', '\\end{bmatrix}'). \
            replace('\\toprule', '').replace('\\midrule', '').replace('\\bottomrule', ''). \
            replace('\\{c\\}', '')

        st.latex(
            r"""
            \begin{{bmatrix}}
            Z_1 \\
            Z_2
            \end{{bmatrix}} = {projection_matrix}^\top
            {sel_feats}
            """.format(projection_matrix=projection_matrix, sel_feats=sel_feats)
        )

    tabs = st.tabs(["Feature Distribution", "Performance Distribution", "Best Algorithm", "SVM Model Prediction", "SVM Selection", "MATILDA Information"])

    with tabs[0]:
        st.subheader("Feature Distribution")
        feature = st.selectbox("Feature", d_features.columns[1:])
        feature_data = st.radio("Feature Data", ["processed", "raw"])
        if feature_data == "raw":
            d_features = pd.read_csv(os.path.join("temp_extracted_files", experiment, "feature_raw.csv"))
        else:
            d_features = pd.read_csv(os.path.join("temp_extracted_files", experiment, "feature_process.csv"))

        fig = plot_feature_distribution(d_features, d_coords, feature)
        st.plotly_chart(fig, use_container_width=True)
        download_feat = st.button("Download Plot", key="feature_plot")
        if download_feat:
            st.write("Downloading plot...")
            fig.update_layout(width=600, height=600)
            fig.write_image(os.path.join("temp_extracted_files", experiment, f"feature_{feature}_{feature_data}_distribution.png"), scale=3)

    with tabs[1]:
        st.subheader("Performance Distribution")
        algorithm = st.selectbox("Algorithm", d_algorithm_raw.columns[1:])
        algorithm_data = st.radio("Algorithm Data", ["raw", "processed"])
        if algorithm_data == "raw":
            d_algorithm = d_algorithm_raw
        else:
            d_algorithm = d_algorithm_process

        fig_algo_perf = plot_performance_distribution(d_algorithm, d_coords, algorithm)
        st.plotly_chart(fig_algo_perf, use_container_width=True)

        st.subheader("Binary Distribution")
        d_algorithm_binary = pd.read_csv(os.path.join("temp_extracted_files", experiment, "algorithm_bin.csv"))
        fig_algo_bin = plot_performance_distribution(d_algorithm_binary, d_coords, algorithm, binary_scale=True)
        st.plotly_chart(fig_algo_bin, use_container_width=True)

        download_perf = st.button("Download Plot", key="performance_plot")
        if download_perf:
            st.write("Downloading plot...")
            fig_algo_perf.update_layout(width=600, height=600)
            fig_algo_bin.update_layout(width=600, height=600)
            fig_algo_perf.write_image(os.path.join("temp_extracted_files", experiment, f"performance_{algorithm}_{algorithm_data}_distribution.png"), scale=3)
            fig_algo_bin.write_image(os.path.join("temp_extracted_files", experiment, f"performance_{algorithm}_{algorithm_data}_binary_distribution.png"), scale=3)

    with tabs[2]:
        st.subheader("Best Algorithm")
        fig = plot_best_algorithm(d_coords, d_best_algo)
        st.plotly_chart(fig, use_container_width=True)
        download_best_algo = st.button("Download Plot", key="best_algo_plot")
        if download_best_algo:
            st.write("Downloading plot...")
            fig.update_layout(width=600, height=600)
            fig.update_layout(legend=dict(orientation="h", xanchor="center", x=0.5, y=-0.2))
            fig.write_image(os.path.join("temp_extracted_files", experiment, "best_algorithm_distribution.png"), scale=3)

    with tabs[3]:
        st.subheader("SVM Selection")
        algorithm_svm = st.selectbox("Algorithm (SVM)", algos)
        show_bounds_selector = st.checkbox("Show Footprints (SVM Selection)")

        if show_bounds_selector:
            fig_svm = plot_svm_selection_single_algo(d_coords, d_svm_preds, algorithm_svm, "temp_extracted_files/" + experiment, show_footprints=True)
        else:
            fig_svm = plot_svm_selection_single_algo(d_coords, d_svm_preds, algorithm_svm, "temp_extracted_files/" + experiment)

        st.plotly_chart(fig_svm, use_container_width=True)
        download_svm = st.button("Download Plot", key="svm_plot")
        if download_svm:
            st.write("Downloading plot...")
            fig_svm.update_layout(width=600, height=600)
            fig_svm.write_image(os.path.join("temp_extracted_files", experiment, f"svm_selection_{algorithm_svm}.png"), scale=3)

    with tabs[4]:
        st.subheader("SVM Selection")
        show_bounds = st.checkbox("Show Footprints")
        fig_svm_selector = plot_svm_selector(d_coords, d_svm_preds, d_svm, experiment_dir="temp_extracted_files/" + experiment, show_footprints=show_bounds)
        st.plotly_chart(fig_svm_selector, use_container_width=True)
        download_svm_selector = st.button("Download Plot", key="svm_selector_plot")
        if download_svm_selector:
            st.write("Downloading plot...")
            fig_svm_selector.update_layout(legend=dict(orientation="h", xanchor="center", x=0.5, y=-0.2))
            fig_svm_selector.update_layout(width=600, height=600)
            fig_svm_selector.write_image(os.path.join("temp_extracted_files", experiment, "svm_selection.png"), scale=3)

    with tabs[5]:
        st.subheader("Model Information")
        with open(os.path.join("temp_extracted_files", experiment, "options.json")) as f:
            options = json.load(f)
        st.json(options, expanded=False)

else:
    st.error("Please upload and process the data on the main page.")
