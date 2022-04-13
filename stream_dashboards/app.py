
from matplotlib import container
import streamlit as st
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

#make contianer

header=st.container()
data_sets=st.container()

features=st.container()
model_traing=st.container()


with header:
    st.title('kashti ki app')
    st.text('in this project we will work on kashti data')

with data_sets:
    st.header("kashti doob gaye")
    st.text('data set header')
    df=sns.load_dataset('titanic')
    df=df.dropna()
    st.write(df.head(2))
    st.subheader("samba katne adme te")
    st.bar_chart(df['sex'].value_counts())

    # other plots
    st.header("class ke hesab se order")
    st.bar_chart(df['class'].value_counts())

    #other plot age

    st.bar_chart(df['age'].sample(10))



with features:
    st.header("these are app feature")
    st.text('variables foof')

    st.markdown("1. ## **Feature 1:** This will tell us pata nahi")
    

with model_traing:
    st.header("kasti walo ka kia bana = model traing")
    st.text('traing stage')
    # making columns

    input,display=st.columns(2)
    max_depth=input.slider("how many people do ypou know",min_value=10,max_value=100,value=20,step=5)


n_estimators=input.selectbox("How may tree : ",options=[50,100,200,300,'No limit'])

#adding list of feature
input.write(df.columns)

#input feature from user
input_features=input.text_input("which feature are should use")


# machine learning model

model=RandomForestRegressor(max_depth=max_depth,n_estimators=n_estimators)

#if else condition 

if n_estimators == 'No limit':
    model=RandomForestRegressor(max_depth=max_depth)

else:
  model=  RandomForestRegressor(max_depth=max_depth,n_estimators=n_estimators)

#define inut and out feature

x=df[[input_features]]
y=df[['fare']]

# fit a model

model.fit(x,y)
pred=model.predict(y)


#display metrics

display.subheader("Mean absolute error of models:")
display.write(mean_absolute_error(y,pred))
display.subheader("Mean squared is error of models:")
display.write(mean_squared_error(y,pred))
display.subheader("R2 score of models:")
display.write(r2_score(y,pred))
