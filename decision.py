import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
my_data = pd.read_csv("drugdecision.csv", delimiter=",")
my_data[0:5]
#Remove the column containing the target name since it doesn't contain numeric values.
X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
X[0:5]
from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1]) 


le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3]) 

X[0:5]
#Now we can fill the target variable.
y = my_data["Drug"]
y[0:5]

#Setting up the Decision Tree
#We will be using train/test split on our decision tree. Let's import train_test_split from sklearn.cross_validation.
from sklearn.model_selection import train_test_split
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)
print(X_trainset)
print(y_trainset)
print(X_testset)
print(y_testset)
#We will first create an instance of the DecisionTreeClassifier called drugTree.
#Inside of the classifier, specify criterion="entropy" so we can see the information gain of each node.
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree # it shows the default parameters
#Entropy is the measure of uncertainty of a random variable, it characterizes the impurity of an arbitrary collection of examples. 
#The higher the entropy the more the information content. The entropy typically changes when we use a node in a decision tree to partition the training instances into smaller subsets.
#Next, we will fit the data with the training feature matrix X_trainset and training response vector y_trainset
drugTree.fit(X_trainset,y_trainset)
#Prediction
#Let's make some predictions on the testing dataset and store it into a variable called predTree.
predTree = drugTree.predict(X_testset)
#You can print out predTree and y_testset if you want to visually compare the prediction to the actual values.
print (predTree [0:5])
print (y_testset [0:5])
#Evaluation
#Next, let's import metrics from sklearn and check the accuracy of our model.
from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))
#accuracy without sklearn
def cal_accuracy(y_test, y_pred): 
      
    print("Confusion Matrix: ", 
        confusion_matrix(y_testset, y_trainset)) 
      
    print ("Accuracy : ", 
    accuracy_score(y_testset,y_trainset)*100) 
      
    print("Report : ", 
    classification_report(y_testset, y_trainset)) 
