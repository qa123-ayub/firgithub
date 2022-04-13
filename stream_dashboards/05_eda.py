from json import load
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling  import st_profile_report

# webapp ka title
st.markdown('''
# **EDA web application**
this app is developed by codanics youtube channel called **EDA app**


''')

#how to upload a file from pc

with st.sidebar.header("Upload your dataset (.csv)"):
    uploaded_file=st.sidebar.file_uploader("Uplod your file ",type=['csv'])
    df=sns.load_dataset('titanic')
    st.sidebar.markdown("[Example CSV file](https://www.kaggle.com/louise2001/quantum-physics-articles-on-arxiv-1994-to-2009/download)")


#profiling report

if uploaded_file is not None:
    @st.cache
    def load_csv():
        csv=pd.read_csv(uploaded_file)
        return csv
    df=load_csv()
    pr=ProfileReport(df,explorative=True)
    st.header('**INput DF**')
    st.write(df)
    st.write('---')
    st.header("**Profiling report with pandas**")
    st_profile_report(pr)

else:
    st.info("AWaiting for csv file ,upload kar b do ab ya kaam nai lena ")
    if st.button("Press to use example data"):
        def load_data():
            a=pd.DataFrame(np.random.rand(100,5),
            columns=['age','banana','codanics','Deutchland','Ear']
            )
            return a
    df=load_data()
    pr=ProfileReport(df,explorative=True)
    st.header('**INput DF**')
    st.write(df)
    st.write('---')
    st.header("**Profiling report with pandas**")
    st_profile_report(pr)
        


