from django.shortcuts import render
# from django.http import HttpResponse

# import ML_model
import pandas as pd
import numpy as np
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.linear_model import *
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# from xgboost import XGBClassifier
sc = StandardScaler()
modelObjects = []
def preProcessing(dataset):
    sn.heatmap(dataset.corr(), annot=True, cmap='viridis', linewidths='2', linecolor='black', fmt='.2g')
    x = dataset.iloc[:, :-1].values
    ye = dataset.iloc[:, -1].values
    x_trainset, x_testset, y_trainset, y_testset = train_test_split(x, ye, test_size=0.3, random_state=2, stratify=ye)
    x_trainset = sc.fit_transform(x_trainset)
    x_testset = sc.transform(x_testset)
    return x_trainset, y_trainset, x_testset, y_testset
# model 1 : Logistic Regression
def classification_model1(x_training, y_training,x_testing,y_testing):
    model1 = LogisticRegression(random_state=1)
    model1.fit(x_training, y_training)
    y_pred1 = model1.predict(x_testing)
    modelObjects.append(model1)
    print(classification_report(y_testing, y_pred1))
    cm = confusion_matrix(y_testing, y_pred1)
    print(cm)
    A1= accuracy_score(y_testing, y_pred1)
    print(A1)
    return y_pred1
# model 2 : KNeightbors classification
def classification_model2(x_training, y_training,x_testing,y_testing):
    model2 = KNeighborsClassifier()
    model2.fit(x_training, y_training)
    y_pred2 = model2.predict(x_testing)
    modelObjects.append(model2)
    print(classification_report(y_testing, y_pred2))
    A2= accuracy_score(y_testing, y_pred2)
    print(A2)
    cm = confusion_matrix(y_testing, y_pred2)
    print(cm)
    return y_pred2
# model3 : support vector classifier
def classification_model3(x_training, y_training,x_testing,y_testing):
    model3 = SVC(random_state=1)
    model3.fit(x_training, y_training)
    y_pred3 = model3.predict(x_testing)
    modelObjects.append(model3)
    print(classification_report(y_testing, y_pred3))
    cm = confusion_matrix(y_testing, y_pred3)
    print(cm)
    A3=accuracy_score(y_testing, y_pred3)
    print(A3)
    return y_pred3
# model 4 : Gaussian Naive Bayes Classifier
def classification_model4(x_training, y_training,x_testing,y_testing):
    model4 = GaussianNB()
    model4.fit(x_training,y_training)
    y_pred4 = model4.predict(x_testing)
    modelObjects.append(model4)
    print(classification_report(y_testing, y_pred4))
    cm = confusion_matrix(y_testing, y_pred4)
    print(cm)
    A4= accuracy_score(y_testing, y_pred4)
    print(A4)
    return y_pred4
# model5 : DecisionTreeClassifier
def classification_model5(x_training, y_training,x_testing,y_testing):
    model5 = DecisionTreeClassifier(random_state=1)
    model5.fit(x_training, y_training)
    y_pred5 = model5.predict(x_testing)
    modelObjects.append(model5)
    print(classification_report(y_testing, y_pred5))
    cm = confusion_matrix(y_testing, y_pred5)
    print(cm)
    A5= accuracy_score(y_testing, y_pred5)
    print(A5)
    return y_pred5
# model6 : RandomForestClassifier
def classification_model6(x_training, y_training,x_testing,y_testing):
    model6 = RandomForestClassifier(random_state=1)
    model6.fit(x_training, y_training)
    y_pred6 = model6.predict(x_testing)
    modelObjects.append(model6)
    print(classification_report(y_testing, y_pred6))
    cm = confusion_matrix(y_testing, y_pred6)
    print(cm)
    A6= accuracy_score(y_testing, y_pred6)
    print(A6)
    return y_pred6
#model7 : XGBoost classifier
# def classification_model7(x_training, y_training,x_testing,y_testing):
#     model7 = XGBClassifier(random_state=1)
#     model7.fit(x_training, y_training)
#     y_pred7 = model7.predict(x_testing)
#     modelObjects.append(model7)
#     print(classification_report(y_testing, y_pred7))
#     cm = confusion_matrix(y_testing, y_pred7)
#     print(cm)
#     A7= accuracy_score(y_testing, y_pred7)
#     print(A7)
#     return y_pred7
def majorityvotinglist(x_training, y_training, x_testing, y_testing):
    Result = []
    k = []
    k.append(classification_model1(x_training, y_training, x_testing, y_testing))
    k.append(classification_model2(x_training, y_training, x_testing, y_testing))
    k.append(classification_model3(x_training, y_training, x_testing, y_testing))
    k.append(classification_model4(x_training, y_training, x_testing, y_testing))
    k.append(classification_model5(x_training, y_training, x_testing, y_testing))
    k.append(classification_model6(x_training, y_training, x_testing, y_testing))
    # k.append(classification_model7(x_training, y_training, x_testing, y_testing))
    for i in range(len(k[0])):
        zero_count = 0
        one_count = 0
        for j in range(len(k)):
            if (k[j][i] == 0):
                zero_count += 1
            else:
                one_count += 1

        if one_count > zero_count:
            Result.append(1)
        else:
            Result.append(0)
    cm = confusion_matrix(y_testing, Result)
    print(cm)
    print(accuracy_score(y_testing, Result))
    return Result
def predictIndividualCase(model,x_test):
    return int(model.predict(x_test))
def predictUserData(x_test):
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (1, 13))
    x_test = sc.transform(x_test)
    allmodelresult = []
    for model in modelObjects:
        allmodelresult.append(predictIndividualCase(model, x_test))
    one = 0
    zero = 0
    for i in allmodelresult:
        if i == 0:
            zero += 1
        else:
            one += 1
    print(allmodelresult)
    if (one >= zero):
        return 1
    return 0
def fileio(L):
    file1 = open("MyFile.txt","w+")
    file1.writelines(str(L))
    file1.close()
def driver():
    dataset = pd.read_csv('heart.csv')
    x_trainset, y_trainset, x_testset, y_testset = preProcessing(dataset)
    y_pred = majorityvotinglist(x_trainset, y_trainset, x_testset, y_testset)
    fileio(y_pred)
    return
def get_predictions(singleTestData):
    driver()
    return predictUserData(singleTestData)
# singleTestData = [47,1,2,108,243,0,1,152,0,0,2,0,2]
# print(getresult(singleTestData))
# print(get_predictions(singleTestData))

# Create your views here.
def home(request):
    return render(request, 'home.html')
    
conversion=[]


def get_conversion(name, age, sex, cp, restbp, chol, fbs, restecg, maxhr, exang, oldpeak, slope, ca, thal):

    ### code here .... input to ML ###
    conversion.append(int(age))
    if sex == "male":
        conversion.append(1)
    else:
        conversion.append(0)
    if cp == "typical angina":
        conversion.append(1)
    elif cp == "atypical angina":
        conversion.append(2)
    elif cp== "non-anginal pain":
        conversion.append(3) 
    else:
        conversion.append(4)
    conversion.append(restbp)
    conversion.append(chol)
    if fbs == "yes":
        conversion.append(1)
    else:
        conversion.append(0) 
    conversion.append(int(restecg))
    conversion.append(int(maxhr))
    if exang == "yes":
        conversion.append(1)
    else:
        conversion.append(0)
    conversion.append(float(oldpeak))
    if slope == "upsloping":
        conversion.append(1)
    elif slope == "flat":
        conversion.append(2)
    else:
        conversion.append(3)
    conversion.append(int(ca))
    if thal == "normal":
        conversion.append(3)
    elif thal == "fixed defect":
        conversion.append(6)
    else:
        conversion.append(7)
    # conversion[12]= int(thal)
    print(conversion)  
    result = get_predictions(conversion)  
    
    # result = get_predictions(T) # machine learning
    ### code here.....  output from ML conversion ###

    if result == 0:
        
        return("Hello "+name+"! Your HEART SEEMS HEALTHY based on the details you entered, if you feel you might have a heart disease. It's better to consult a doctor. ")

    else:
        return("Hello "+name+"! You are at a RISK OF GETTING A HEART DISEASE. Please consult a doctor for the further details.")


    # return 



def result(request):
    name = request.GET['name']
    age = request.GET['age']
    sex = request.GET['sex']
    cp = request.GET['cp']
    restbp = int(request.GET['restbp'])
    chol = int(request.GET['chol'])
    fbs = request.GET['fbs']
    restecg = request.GET['restecg']
    maxhr = int(request.GET['maxhr'])
    exang = request.GET['exang']
    oldpeak = float(request.GET['oldpeak'])
    slope = request.GET['slope']
    ca = request.GET['ca']
    thal = int(request.GET['thal'])
    
    name = name.upper()
    result= get_conversion(name, age, sex, cp, restbp, chol, fbs, restecg, maxhr, exang, oldpeak, slope, ca, thal)

    return render(request, 'result.html', {'result': result})

    
