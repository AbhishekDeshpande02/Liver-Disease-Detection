import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
patients=pd.read_csv('D:/indian_liver_patient.csv')

patients.head()
patients.shape
patients['Gender']=patients['Gender'].apply(lambda x:1 if x=='Male' else 0)
patients.head()

patients['Dataset'].value_counts().plot.bar(color='blue')
patients.isnull().sum()
patients['Albumin_and_Globulin_Ratio'].mean()
patients=patients.fillna(0.94)
patients.isnull().sum()
sns.set_style('darkgrid')
plt.figure(figsize=(25,10))
patients['Age'].value_counts().plot.bar(color='darkviolet')
plt.rcParams['figure.figsize']=(10,10)
sns.pairplot(patients,hue='Gender')
sns.pairplot(patients)
f, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x="Albumin", y="Albumin_and_Globulin_Ratio",color='mediumspringgreen',data=patients);
plt.show()
plt.figure(figsize=(8,6))
patients.groupby('Gender').sum()["Total_Protiens"].plot.bar(color='coral')
plt.figure(figsize=(8,6))
patients.groupby('Gender').sum()['Albumin'].plot.bar(color='midnightblue')
plt.figure(figsize=(8,6))
patients.groupby('Gender').sum()['Total_Bilirubin'].plot.bar(color='fuchsia')

corr=patients.corr()
plt.figure(figsize=(20,10)) 
sns.heatmap(corr,cmap="Greens",annot=True)
plt.show()

from sklearn.model_selection import train_test_split
patients.columns
X=patients[['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin',
       'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
       'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin',
       'Albumin_and_Globulin_Ratio']]
y=patients['Dataset']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=123)


logistic_regression_classifier = LogisticRegression(max_iter=1000, random_state=42)  # You can experiment with different hyperparameters

# Step 3: Train the Logistic Regression model
logistic_regression_classifier.fit(X_train, y_train)

# Step 4: Predict using the trained model
y_pred = logistic_regression_classifier.predict(X_test)

# Step 5: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print the results
print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")


# svm

svm_classifier = SVC(kernel='rbf', C=1.0, gamma='auto')  # You can experiment with different kernels and hyperparameters

# Step 3: Train the SVM model
svm_classifier.fit(X_train, y_train)

# Step 4: Predict using the trained model
y_pred = svm_classifier.predict(X_test)

# Step 5: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print the results
print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")

#Random-Forest

random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)  # You can experiment with different hyperparameters

# Step 3: Train the Random Forest model
random_forest_classifier.fit(X_train, y_train)

# Step 4: Predict using the trained model
y_pred = random_forest_classifier.predict(X_test)

# Step 5: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print the results
print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")
