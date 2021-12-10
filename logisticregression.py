import pandas as pd
#read data
data =pd.read_csv("filename")
data.head()
Y=data.iloc[:,[8]]
X=data.iloc[:,[1,2,3,4,5,6,7]]
Y
X
# Import module to split dataset
from sklearn.model_selection import train_test_split
# Split data set into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
# Import module for fitting
from sklearn.linear_model import LogisticRegression
# Create instance (i.e. object) of LogisticRegression
logmodel = LogisticRegression()
# Fit the model using the training data
# X_train -&gt; parameter supplies the data features
# y_train -&gt; parameter supplies the target labels
logmodel.fit(X_train, Y_train)
Y_pred = logmodel.predict(X_test)
Y_pred
#testing the accuracy
from sklearn.metrics import accuracy_score

accuracy=accuracy_score(Y_test,Y_pred)

print(str(accuracy*100)+&quot;% accuracy&quot;)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))