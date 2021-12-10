import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
#read data
data = df=pd.read_csv("filename")
#setting up the data

y=data.iloc[:,[8]]
x=data.iloc[:,[1,2,3,4,5,6,7]]
#preprocessing the data

#le = preprocessing.LabelEncoder()
#y=le.fit_transform(y)

#splitting the data into train and test

X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)
#training the classifier
clf=tree.DecisionTreeClassifier(criterion=&#39;gini&#39;,min_samples_split=30,splitter=&quot;best&quot;)
clf=clf.fit(X_train,Y_train)

#predicting

y_pred=clf.predict(X_test)

#testing the accuracy

accuracy=accuracy_score(Y_test,y_pred)
print(str(accuracy*100)+&quot;% accuracy&quot;)
from sklearn.metrics import classification_report
print(classification_report(Y_test,y_pred))