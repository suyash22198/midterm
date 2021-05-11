import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding', False)
# Load the pickled model
pickle_in = open("suyash.pkl","rb")
model=pickle.load(pickle_in)
dataset= pd.read_csv('data.csv')
X = dataset.iloc[:,0:8].values

  # Taking care of missing data
#handling missing data (Replacing missing data with the mean value)  
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.NAN, strategy= 'mean', fill_value=None, verbose=1, copy=True)
#Fitting imputer object to the independent variables x.   
imputer = imputer.fit(X[:, 2:7])
#Replacing missing data with the calculated mean value  
X[:, 2:7]= imputer.transform(X[:, 2:7])


# Encoding Categorical data:
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


def predict_note_authentication(Gender,Glucose,BP,SkinThickness,Insulin,BMI,PedigreeFunction,Age):
  output= model.predict(sc.transform([[Gender,Glucose,BP,SkinThickness,Insulin,BMI,PedigreeFunction,Age]]))
  print("modal is predicted ",output)
  if output==[0]:
    prediction="Decision tree modal is predicted whether the person has disease  "
   

  if output==[1]:
    prediction="Decision tree modal is predicted whether the person has NOT any disease  "
    
    
  print(prediction)
  return prediction
def main():
    
    html_temp = """
   <div class="" style="background-color:black;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
   <center><p style="font-size:30px;color:white;margin-top:10px;">Department of Computer Engineering</p></center> 
   <center><p style="font-size:20px;color:white;margin-top:10px;">MID TERM 1 pactice by piet18cs141 Suyash Sharma</p></center> 
   </div>
   </div>
   </div>
   """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.header("Decision tree modal is predicted whether the person has disease  Or NOT")
    Gender = st.number_input('Insert gender 0 for male and 1 for female',0,1)
    Glucose = st.number_input('Enter  Glucose lavel ',10,300)
    BP = st.number_input('Enter your BP lavel',0,100)
    SkinThickness = st.number_input('Enter skinthickness ',0,100)
    Insulin = st.number_input('Enter insulin level ',0,500)
    BMI = st.number_input('Enter BMI ',0,100)
    PedigreeFunction = st.number_input('Enter your pedigree Function ',0.0,1.0)
    Age= st.number_input('Enter Age',5,100)
   
    
    resul=""
    if st.button("Predict"):
      result=predict_note_authentication(Gender,Glucose,BP,SkinThickness,Insulin,BMI,PedigreeFunction,Age)
      st.success('Model has predicted {}'.format(result))
    if st.button("About"):
      st.subheader("Developed by Suyash Sharma 1st mid term ")
      st.subheader("C-Section,PIET")

if __name__=='__main__':
  main()
   
