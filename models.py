from importlib import import_module
from re import T
import numpy as np
import pandas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
patients=pd.read_csv('D:/indian_liver_patient.csv')#change with the location of your .csv file

patients.head()
patients.shape
patients['Gender']=patients['Gender'].apply(lambda x:1 if x=='Male' else 0)
patients.head()
patients['Gender'].value_counts().plot.bar(color='peachpuff')
patients['Dataset'].value_counts().plot.bar(color='blue')
patients.isnull().sum()
patients['Albumin_and_Globulin_Ratio'].mean()
patients=patients.fillna(0.94)
patients.isnull().sum()



from sklearn.model_selection import train_test_split

patients.columns

X=patients[['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin',
       'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
       'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin',
       'Albumin_and_Globulin_Ratio']]
y=patients['Dataset']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=123)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

from sklearn.model_selection import KFold, cross_val_score
kfold = KFold(shuffle= True,n_splits=5,random_state=42)
logmodel = LogisticRegression(C=1, penalty='l2')
results = cross_val_score(logmodel, X_train,y_train,cv = kfold)
print(results)
print("Accuracy:",results.mean()*100)


# Train the Logistic Regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X, y)

def predict_disease():
    Age = float(input("Enter Age: "))
    Gender = input("Enter Gender (Male/Female): ")
    Total_Bilirubin = float(input("Enter Total_Bilirubin:"))
    Direct_Bilirubin = float(input("Enter Direct_Bilirubin:"))
    Alkaline_Phosphotase = int(input("Enter Alkaline_Phosphotase:"))
    Alamine_Aminotransferase = int(input("Enter Alamine_Aminotransferase:"))
    Aspartate_Aminotransferase = int(input("Enter Aspartate_Aminotransferase:"))
    Total_Protiens = float(input("Enter Total_Protiens:"))
    Albumin = float(input("Enter Albumin:"))
    Albumin_and_Globulin_Ratio = float(input("Enter Albumin_and_Globulin_Ratio:"))
   

    # Create a DataFrame with user input
    user_data = pd.DataFrame({
        'Age': [Age],
        'Gender': [1 if Gender == 'Male' else 0],
        'Gender' : [1 if Gender == 'Female' else 0],
        'Total_Bilirubin': [Total_Bilirubin],
        'Direct_Bilirubin': [Direct_Bilirubin],
        'Alkaline_Phosphotase': [Alkaline_Phosphotase],
        'Alamine_Aminotransferase': [Alamine_Aminotransferase],
        'Aspartate_Aminotransferase' : [Aspartate_Aminotransferase],
        'Total_Protiens' : [Total_Protiens],
        'Albumin' : [Albumin],
        'Albumin_and_Globulin_Ratio' : [Albumin_and_Globulin_Ratio],
    })

    # Make the prediction
    prediction = model.predict(user_data)

    if prediction[0] == 1:
        print("You may have liver disease.")
    else:
        print("You may not have liver disease.")
        
if __name__ == "__main__":
    predict_disease()
