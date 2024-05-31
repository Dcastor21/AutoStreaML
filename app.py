import streamlit as st
import pandas as pd
import os

import pandas_profiling 
from streamlit_pandas_profiling import st_profile_report


from pycaret.classification import setup, compare_models, pull, save_model

with st.sidebar:
    st.image("assets\img\AutostreamML.png")
    st.title(" AutostreamML ")
    choice = st.radio("Navigation", ["Upload", "Profiling", "ML", "Download"])
    st.info("this Application Builds an automated ML pipeline using Streamlit, PyCaret.")

if os.path.exists("sourcedata.csv"):
    df=pd.read_csv("sourcedata.csv", index_col=None)

if choice == "Upload":
    st.title("Upload Your Data for Modeling!")
    file= st.file_uploader("Upload Your Dataset Here")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("sourcedata.csv", index=None )
        st.dataframe(df)
if  choice == "Profiling":
    st.title("Automated Exploratory Data Analysis")
    profile_report(profile_report)
if  choice == "ML":
    st.title("Automated Machine Learning")
    target = st.selectbox("Select Your Target", df.columns)
    if st.button("Train Model"):
        setup(df, target=target, silent=True)
        setup_df = pull()
        st.info("This is the ML Experiment Settings")
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.info("This Is the ML Model")
        st.dataframe(compare_df)
        best_model
        save_model(best_model,"best Model")
    

if  choice == "Download":
    with open ("best_model.pkl", 'rb') as f:
        st.download_button("Download the Model", f, "trained_model.pkl")