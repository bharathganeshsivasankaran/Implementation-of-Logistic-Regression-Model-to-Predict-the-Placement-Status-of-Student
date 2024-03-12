# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: BHARATHGANESH S
RegisterNumber:  212222230022
*/
```
```
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

### Output:
### Read data
![image](https://github.com/bharathganeshsivasankaran/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119478098/94fa0320-134c-4ad2-9db4-a0510704ca40)
### Droping unwanted coloumn
![Screenshot 2024-03-12 093149](https://github.com/bharathganeshsivasankaran/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119478098/6ea44d06-c218-462a-a5f6-1eaf26e7ff2c)
### Persence of null value
![image](https://github.com/bharathganeshsivasankaran/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119478098/50939e11-6d6c-4529-8878-51772941d477)
## Duplicated value
![image](https://github.com/bharathganeshsivasankaran/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119478098/884d293a-4f63-41cd-a812-9e983160d49a)
### Data encoding
![image](https://github.com/bharathganeshsivasankaran/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119478098/d8a634b5-e51e-43f5-92c9-942fa4ef8605)
### X data
![image](https://github.com/bharathganeshsivasankaran/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119478098/cae8199f-a584-421f-9e49-ebde7fd6206e)
### y data
![image](https://github.com/bharathganeshsivasankaran/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119478098/8c9ef610-8c78-4424-bb03-24130f3845c9)
### confusion matrix
![image](https://github.com/bharathganeshsivasankaran/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119478098/88faeef0-5844-45cd-a515-2a32e6e8c6cd)
# classification report
![image](https://github.com/bharathganeshsivasankaran/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119478098/e31f1379-3f62-4af1-a4a8-c6e068eade4c)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
