# Company-Trend-analysis-project-docs
This repository contains all the documentations used in creating the trend analysis machine learning project
Project writeup:
[Company success prediction.docx](https://github.com/user-attachments/files/16811915/Company.success.prediction.docx)

Powerpoint Presentation:

[Machine Learning Prediction of Companies’ Business Success ppt.pptx](https://github.com/user-attachments/files/16811922/Machine.Learning.Prediction.of.Companies.Business.Success.ppt.pptx)

Dataset:

[CompanyDataset.csv](https://github.com/user-attachments/files/16811924/CompanyDataset.csv)

Code:

MACHINE LEARNING PROJECT TO DETERMINE COMPANY SUCCESS OR FAILURE

#IMPORTING NECESSARY LIBRARIES:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.offline import iplot , plot
from plotly.subplots import make_subplots

LOAD THE DATA

data=pd.read_csv("CompanyDataset.csv")


EXPLORATORY DATA ANALYSIS

data.info()

data.columns

data.shape

data.describe()

data.sample(5)

data.isna().sum()

data.isna().any()

data.duplicated().any()

data['Bankrupt?'].unique()

DATA VISUALIZATION

sns.displot(data['Bankrupt?'])

:DATA PREPROCESSING

x=data.drop('Bankrupt?',axis=1)
y=data['Bankrupt?']

from sklearn.preprocessing import LabelEncoder
encoder_x=LabelEncoder()

for col in x.columns:
    x[col]=encoder_x.fit_transform(x[col])

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3,shuffle=True,random_state=42)

print("x_train shape = ", x_train.shape)
print("y_train shape = ", y_train.shape)
print("x_test shape = ", x_test.shape)
print("y_test shape = ", y_test.shape)

shapes = {
    'X_train': x_train.shape[0],
    'y_train': y_train.shape[0],
    'X_test': x_test.shape[0],
    'y_test': y_test.shape[0]
}
plt.figure(figsize=(15, 6))
plt.bar(shapes.keys(), shapes.values())
plt.xlabel('Datasets')
plt.ylabel('Number of instances')
plt.title('Distribution of Training and Validation Sets')
plt.show()

DATA SCALING

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

x_train=scaler.fit_transform(x_train)
x_test=scaler.fit_transform(x_test)

MODEL 1:LOGISTIC REGRESSION

from sklearn.linear_model import LogisticRegression
lr_model=LogisticRegression() # call model
lr_model.fit(x_train,y_train)

y_pred=lr_model.predict(x_test)
y_pred

IMPLEMENTING A CONFUSION MATRIX

from sklearn.metrics import confusion_matrix

con= confusion_matrix(y_test,y_pred) # Evaluation of Model Performance & Sensitivity and Specificity Analysis

sns.heatmap(con, annot=True, cmap='viridis', cbar=True) # heatmap for Matrix Data Representation

from sklearn.metrics import classification_report # for Precision and Recall Analysis

print("classification_report is ",classification_report(y_test ,y_pred))

MODEL 2: SUPPORT VECTOR MACHINE(SVM)

from sklearn.svm import SVC
from sklearn.svm import SVC ## call model

svm_model =SVC()

svm_model.fit(x_train, y_train)

y_pred = svm_model.predict(x_test)
y_pred

print("classification_report is ",classification_report(y_test ,y_pred))

#output

![confusion matrix](https://github.com/user-attachments/assets/25a759fc-c335-443c-bd68-90b2eb7bb8aa)

![Accuracy table](https://github.com/user-attachments/assets/b0ef2974-1c01-4728-b2db-fd71e2259e8b)
![accuracy table for Support vector machine](https://github.com/user-attachments/assets/3dba1b9e-b148-4f4d-9f75-bdc2cf650bd6)


Compiled by Holiness Supremacy Mhlanga

