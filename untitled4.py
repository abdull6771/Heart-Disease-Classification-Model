import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import accuracy_score,f1_score,precision_score
import seaborn as sns

# Importing Dataset
Data_set= pd.read_csv("heart-disease.csv")
Data_set.info()
X= Data_set.drop(columns="target",axis=1)
Y = Data_set["target"]
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=(123))

models = [RandomForestClassifier(),DecisionTreeClassifier(),GaussianNB(),SVC(),
         KNeighborsClassifier()]
def comparing():
    for model in models:
        model.fit(x_train,y_train)
        predict = model.predict(x_test)
        accu = accuracy_score(y_test, predict)
        print("The accuracy Score for the ",model,"=",accu)
comparing()

RDD = RandomForestClassifier()
params ={
    "n_estimators":[40,60,80,100,150],
    "max_depth":[3,6,9,12,15],
    "min_samples_split":[2,3,4,5,6]
    }
GRD = GridSearchCV(RDD, params,cv=(5))
GRD.fit(X,Y)
GRD.best_estimator_