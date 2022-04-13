import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#Heading od app
st.write("""

# Explore different ML models and datasets
which one is best ? "See" : Unknown word.
""")

dataset_name=st.sidebar.selectbox(
    "Select Dataset",
    ('iris','Breast Cancer','Wine')

)


classifier_name=st.sidebar.selectbox(
 'Select classifier',
 ('KNN','SVM','Random Forest')

)

#import Datasets

def get_dataset(dataset_name):
    data=None
    if dataset_name=='iris':
        data=datasets.load_iris()
    elif dataset_name=='Breast Cancer':
        data=datasets.load_breast_cancer()
    else:
        data=datasets.load_wine()

    #separate the input variables from classes
    x=data.data
    y=data.target
    return x, y


x,y=get_dataset(dataset_name)

st.write('shape of dataset',x.shape)
st.write('number of classes',len(np.unique(y)))



# make a a function to run ml algorithm

def add_parameter(classifier_name):
    params=dict()
    if classifier_name=='SVM':
        C=st.sidebar.slider('C',0.01,10.0)
        params['C']=C

    elif classifier_name=='KNN':
        K=st.sidebar.slider('K',1,15)
        params['K']=K

    else:

        max_depth=st.sidebar.slider('max_depth',2,15)
        params['max_depth']=max_depth #depth of every tree that grow in random forest
        n_estimators=st.sidebar.slider('n_estimators',1,100)
        params['n_estimators'] =n_estimators #number of trees
    return params

   
#


params=add_parameter(classifier_name)

def get_classifier(classifier_name,params):
    if classifier_name=='SVM':
        clf=SVC(C=params['C'])
    elif classifier_name=='KNN':
        clf=KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf=RandomForestClassifier(n_estimators=params['n_estimators']
        ,max_depth=params['max_depth'],random_state=1234)

    return clf


# now call the function
clf=get_classifier(classifier_name,params)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1234)
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)


acc=accuracy_score(y_test,y_pred)

st.write(f'Classifier= {classifier_name}')
st.write(f'Accuracy =' ,acc)


#plotting
pca=PCA(2)
x_projected=pca.fit_transform(x)

x1=x_projected[:,0]
x2=x_projected[:,1]

fig=plt.figure()
plt.scatter(x1,x2,c=y,alpha=0.8,cmap='viridis')
plt.xlabel('principal Componenet 1 ')
plt.ylabel('principal Componenet 2 ')
plt.colorbar()

#plot show

st.pyplot(fig)