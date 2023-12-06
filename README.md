# final-year-project-
#collage project
Importing the Dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier



from sklearn.metrics import accuracy_score
Data Collection and Processing
# loading the csv data to a Pandas DataFrame
Diabetes_data = pd.read_csv('diabetesdata.csv')
# print first 5 rows of the dataset
Diabetes_data.head()
# print last 5 rows of the dataset
Diabetes_data.tail()
# number of rows and columns in the dataset
Diabetes_data.shape
# getting some info about the data
Diabetes_data.info()
# checking for missing values
Diabetes_data.isnull().sum()
# statistical measures about the data
Diabetes_data.describe()
# checking the distribution of Target Variable
Diabetes_data['Target'].value_counts()
1 --> Type 2 Diabetes

0 --> Type 1 Diabetes
Splitting the Features and Target
X = Diabetes_data.drop(columns='Target', axis=1)
Y = Diabetes_data['Target']
print(X)
print(Y)
Splitting the Data into Training data & Test Data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

Model Training
Logistic Regression
# training the LogisticRegression model with Training data
model = LogisticRegression()
model.fit(X_train, Y_train)


# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy on Test data : ', test_data_accuracy)
Decision Tree
dc_model = DecisionTreeClassifier()
dc_model.fit(X_train, Y_train)

#accuracy

dc_test_prediction = dc_model.predict(X_test)
Test_data_accuracy = accuracy_score(Y_test, dc_test_prediction)

print('Test Data Accuracy : ', Test_data_accuracy)
Random Forest
rc_model = RandomForestClassifier()
rc_model.fit(X_train, Y_train)

#accuracy

rc_test_prediction = rc_model.predict(X_test)
Test_data_accuracy = accuracy_score(Y_test, rc_test_prediction)

print('Test Data Accuracy : ', Test_data_accuracy)
SVM
svm_model = SVC()
svm_model.fit(X_train, Y_train)

#accuracy

svm_test_prediction = svm_model.predict(X_test)
Test_data_accuracy = accuracy_score(Y_test, svm_test_prediction)

print('Test Data Accuracy : ', Test_data_accuracy)
K Nearest Neighbors
from sklearn.model_selection import cross_val_score
knn_scores=[]
for k in range(1,16):
  knn_classifier=KNeighborsClassifier(n_neighbors=k)
  score=cross_val_score(knn_classifier,X,Y,cv=10)
  knn_scores.append(score.mean())
plt.plot([k for k in range(1,16)], knn_scores, color='red')
for i in range(1,16):
  plt.text(i, knn_scores[i-1], (i, knn_scores[i-1]))
  plt.xticks([i for i in range(1,16)])
  plt.xlabel('Number of Neighbors(K)')
  plt.ylabel('Scores')
  plt.title('K Neighbors Classifier scores for different k values')
Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
gbc=GradientBoostingClassifier()
gbc.fit(X_train,Y_train)
y_pred6=gbc.predict(X_test)
accuracy_score(Y_test,y_pred6)
print("Gradient Boosting Classifier :",accuracy_score(Y_test,y_pred6)*100)
Logistic Regression
from sklearn.linear_model import LogisticRegression
log=LogisticRegression()
log.fit(X_train,Y_train)
y_pred1=log.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(Y_test,y_pred1)
SVC (Support Vector Classifier)
from sklearn import svm
svm=svm.SVC()
svm.fit(X_train,Y_train)
y_pred2=svm.predict(X_test)
accuracy_score(Y_test,y_pred2)
K Neighbors Classifier
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(X_train,Y_train)
y_pred3=knn.predict(X_test)
accuracy_score(Y_test,y_pred3)
best value for the number of neighbors
score=[]
for k in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,Y_train)
    y_pred=knn.predict(X_test)
    score.append(accuracy_score(Y_test,y_pred))
import matplotlib.pyplot as plt
plt.plot(score)
plt.xlabel("k value")
plt.ylabel("Acc")
plt.show()
Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(X_train,Y_train)
y_pred4=dt.predict(X_test)
accuracy_score(Y_test,y_pred4)
Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(X_train,Y_train)
y_pred5=rf.predict(X_test)
accuracy_score(Y_test,y_pred5)
Pivot table
final_data=pd.DataFrame({'Models':['LR','SVM','KNN','DT','RF','GB'],
                        'ACC':[accuracy_score(Y_test,y_pred1)*100,
                              accuracy_score(Y_test,y_pred2)*100,
                              accuracy_score(Y_test,y_pred3)*100,
                              accuracy_score(Y_test,y_pred4)*100,
                              accuracy_score(Y_test,y_pred5)*100,
                              accuracy_score(Y_test,y_pred6)*100]})
final_data
import seaborn as sns
sns.barplot(final_data['Models'],final_data['ACC'])
 Prediction on New Data
import pandas as pd
new_data =pd.DataFrame({
    'age':52, 
    'Pregnancies':6,
    'Glucose':148,
    'BloodPressure':72,
    'SkinThicknes':35,
    'Insulin':0,
    'BMI':33.6,
    'DiabetesPedigreeFunction':0.627
    
    
    
},index=[0])
new_data
p=gbc.predict(new_data)
if p[0] ==0:
    print("Type 2 Diabetes")
else:
    print("Type 1 Diabetes")
import joblib
joblib.dump(gbc,'model_joblib_Diabetes')
Saved model for Random forest
model=joblib.load('model_joblib_Diabetes')
model.predict(new_data)
Gui for the project
import tkinter
from tkinter import *
from PIL import Image,ImageTk
import joblib
import os
def show_entry_fields():
    p1=int(e1.get())
    p2=int(e2.get())
    p3=int(e3.get())
    p4=int(e4.get())
    p5=int(e5.get())
    p6=int(e6.get())
    p7=float(e7.get())
    p8=float(e8.get())
    model = joblib.load('model_joblib_Diabetes')
    p[0]=model.predict([[p1,p2,p3,p4,p5,p6,p7,p8]])
    
    if p[0] == 0:
        Label(master, text="You have TYPE 2 DIABETES",font=("Arial",18)).grid(row=31)
    else:
        Label(master, text="  You have TYPE 1 DIABETES",font=("Arial",18)).grid(row=31)
    
    
master=Tk()
master.geometry("1400x800")

image1 = Image.open("C:\\Users\\niranjan\\Pictures\\Diabete.PNG")
test = ImageTk.PhotoImage(image1)
label1 = tkinter.Label(image=test)
label1.image = test

label1.place(x=0, y=0)



master.title("Diabetes Type Prediction System")


label = Label(master, text = "  Diabetes Type Prediction System     ",font=("Arial",20)
                          , bg = "black", fg = "white").grid(row=0,columnspan=2)


Label(master, text="Enter Your Age" ,font=("Arial",16)).grid(row=1,sticky='w')
Label(master, text="Enter Value of Child birth weight",font=("Arial",16)).grid(row=2,sticky='w')
Label(master, text="Enter Value of Glucose",font=("Arial",16)).grid(row=3,sticky='w')
Label(master, text="Enter Value of BloodPressure",font=("Arial",16)).grid(row=4,sticky='w')
Label(master, text="Enter Value of SkinThickness",font=("Arial",16)).grid(row=5,sticky='w')
Label(master, text="Enter Value of Insulin",font=("Arial",16)).grid(row=6,sticky='w')
Label(master, text="Enter Value of BMI",font=("Arial",16)).grid(row=7,sticky='w')
Label(master, text="Enter Value of DiabetesPedigreeFunction",font=("Arial",16)).grid(row=8,sticky='w')



e1 = Entry(master)
e2 = Entry(master)
e3 = Entry(master)
e4 = Entry(master)
e5 = Entry(master)
e6 = Entry(master)
e7 = Entry(master)
e8 = Entry(master)

e1.grid(row=1, column=1)
e2.grid(row=2, column=1)
e3.grid(row=3, column=1)
e4.grid(row=4, column=1)
e5.grid(row=5, column=1)
e6.grid(row=6, column=1)
e7.grid(row=7, column=1)
e8.grid(row=8, column=1)

def delete():
    e1.delete(0,END)
    e2.delete(0,END)
    e3.delete(0,END)
    e4.delete(0,END)
    e5.delete(0,END)
    e6.delete(0,END)
    e7.delete(0,END)
    e8.delete(0,END)
    
Button(master, text='Predict',font=("Arial",14),bg="white", command=show_entry_fields).grid()
Button(master, text='Clear',font=("Arial",14),bg="white", command=delete).grid()
master.mainloop()


