# EXN0-4 Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5. Display the results.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: BHARATHGANESH.S
RegisterNumber:  212222230022
*/
```
```
import pandas as pd
data=pd.read_csv("C:\classes\ML\Placement_Data.csv")data.head()
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
## Output:
### Data set:
![image](https://github.com/Ashwinkumar-03/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118663725/e11805b8-d6a0-445d-8547-476b240d5738)

### Salary data:
![image](https://github.com/Ashwinkumar-03/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118663725/8f306f81-7961-4611-89fd-2960de5baf8c)

### Checking the null() function:
![image](https://github.com/Ashwinkumar-03/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118663725/220a7051-d002-45ed-be80-365c01e36180)

### checking Duplicate data:
![image](https://github.com/Ashwinkumar-03/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118663725/4ace3b51-9884-4f11-a515-533f3daacf56)

### data:
![image](https://github.com/Ashwinkumar-03/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118663725/d0bdcc66-b7b0-406f-8296-2444f4cd94ae)

### data status:
![image](https://github.com/Ashwinkumar-03/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118663725/70dd1fb3-215a-42f7-b18d-6d06f24e4bc7)

### y_predict:
![image](https://github.com/Ashwinkumar-03/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118663725/967b11ae-019c-4744-83ce-a9763e6384dd)

### Accuracy value:
![image](https://github.com/Ashwinkumar-03/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118663725/96214719-0852-48b3-a6fc-cf2c12cfb3e3)

### Confusion array:
![image](https://github.com/Ashwinkumar-03/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118663725/de0c8638-53e4-4e09-9518-bfbeb230825d)

### Classification Report:
![image](https://github.com/Ashwinkumar-03/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118663725/39523db3-49e2-464c-b5ad-efd0cc4e1092)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
