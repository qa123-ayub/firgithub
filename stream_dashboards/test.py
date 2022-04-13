import streamlit as st
import seaborn as sns


st.header("This video is brought to by  #babaAmmar")
st.text("kia maxa aya aplogo ko")

st.header("pata nahi kia lekna hea")

st.header("deka kamal")
st.text("lo ge")
df=sns.load_dataset('iris')
st.write(df.head(10))

st.write(df[['species','sepal_length']].head(10))
st.bar_chart(df['sepal_length'])

st.line_chart(df['sepal_length'])