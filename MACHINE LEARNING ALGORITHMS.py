# IN THIS PROJECT, WE USE R LANGUAGE WITH RATTLE TOOLS (YOU NEED TO INSTALL THIS TOOLS IN R ENVIRONMENT LANGUAGE) AND 
#YOU CAN UTILIZE THE PYTHON PROGRAM AS SHOWN BELOW TO FIND RESULTS
#HENRY TRAN


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#input dataset
dt = pd.read_csv("train.csv")

# check missing values
dt.isnull().sum()
# Cleaning dataset
dt['clean_tweet'] = dt['tweet']

import string

def remove_punct(text):
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text

dt['clean_tweet'] = dt['tweet'].apply(lambda x: remove_punct(x))
dt.head(20)
# Cleaning the tweets

 
# Compare the Logistic Regression Model V.S. Base Rate Model V.S. Random Forest Model
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
# Splitting the dataset into the Training set and Test set
#GPA Dataset
# input 
X = dt.iloc[:, [1,2]].values   
print(X)
# output 
y = dt.iloc[:, 0].values 
print(y)
from sklearn.model_selection import train_test_split 

Xtrain, Xtest, ytrain, ytest = train_test_split( 
        X, y, test_size = 0.30, random_state = 0) 



# PCA 
pca = PCA(n_components=2, svd_solver='full')
pca.fit(Xtrain, ytrain)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)


# LINEAR REGRESSION
reg = linear_model.LinearRegression()
reg.fit(Xtrain, ytrain)
print ("\n\n ---Linear Regression Model---")
reg_roc_auc = roc_auc_score(ytest, reg.predict(Xtest))
print ("Linear Regression AUC = %2.2f" % reg_roc_auc)
#print(classification_report(ytest, reg.predict(Xtest)))

# LOGISTICS REGRESSION
logis = LogisticRegression(class_weight = "balanced")
logis.fit(Xtrain, ytrain)
print ("\n\n ---Logistic Model---")
logit_roc_auc = roc_auc_score(ytest, logis.predict(Xtest))
print ("Logistic AUC = %2.2f" % logit_roc_auc)
print(classification_report(ytest, logis.predict(Xtest)))

# Decision Tree Model
dtree = tree.DecisionTreeClassifier(
    #max_depth=3,
    class_weight="balanced",
    min_weight_fraction_leaf=0.01
    )
dtree = dtree.fit(Xtrain,ytrain)
print ("\n\n ---Decision Tree Model---")
dt_roc_auc = roc_auc_score(ytest, dtree.predict(Xtest))
print ("Decision Tree AUC = %2.2f" % dt_roc_auc)
print(classification_report(ytest, dtree.predict(Xtest)))

# Random Forest Model
rf = RandomForestClassifier(
    n_estimators=1000, 
    max_depth=None, 
    min_samples_split=10, 
    class_weight="balanced"
    #min_weight_fraction_leaf=0.02 
    )
rf.fit(Xtrain, ytrain)
print ("\n\n ---Random Forest Model---")
rf_roc_auc = roc_auc_score(ytest, rf.predict(Xtest))
print ("Random Forest AUC = %2.2f" % rf_roc_auc)
print(classification_report(ytest, rf.predict(Xtest)))


# Ada Boost
ada = AdaBoostClassifier(n_estimators=400, learning_rate=0.1)
ada.fit(Xtrain,ytrain)
print ("\n\n ---AdaBoost Model---")
ada_roc_auc = roc_auc_score(ytest, ada.predict(Xtest))
print ("AdaBoost AUC = %2.2f" % ada_roc_auc)
print(classification_report(ytest, ada.predict(Xtest)))

# SVM MODEL
from sklearn.svm import SVC
from sklearn import svm
svc = svm.SVC(kernel='rbf', C=1,gamma=0.1,probability=True).fit(Xtrain,ytrain)
print ("\n\n ---SVM Model---")

from sklearn.metrics import accuracy_score
svc_roc_auc = roc_auc_score(ytest, svc.predict(Xtest))
print ("SVM AUC = %2.2f" % svc_roc_auc)
from sklearn.metrics import classification_report
print(classification_report(ytest, svc.predict(Xtest)))

#Artifical Neural Network model
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X, y)
MLPClassifier(alpha=1e-05, hidden_layer_sizes=(5, 2), random_state=1,
              solver='lbfgs')
print ("\n\n ---Neural Network Model---")
clf_roc_auc = roc_auc_score(ytest, clf.predict(Xtest))
print ("ANN AUC = %2.2f" % clf_roc_auc)
print(classification_report(ytest, clf.predict(Xtest)))


# Create ROC Graph and show AUC

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(ytest, logis.predict_proba(Xtest)[:,1])
#lr_fpr, lr_tpr, thresholds = roc_curve(ytest, reg.predict_proba(Xtest)[:,1])
rf_fpr, rf_tpr, rf_thresholds = roc_curve(ytest, rf.predict_proba(Xtest)[:,1])
dt_fpr, dt_tpr, dt_thresholds = roc_curve(ytest, dtree.predict_proba(Xtest)[:,1])
ada_fpr, ada_tpr, ada_thresholds = roc_curve(ytest, ada.predict_proba(Xtest)[:,1])
svc_fpr, svc_tpr, svc_thresholds = roc_curve(ytest, svc.predict_proba(Xtest)[:,1])
clf_fpr,clf_tpr,clf_thresholds = roc_curve(ytest, clf.predict_proba(Xtest)[:,1])
plt.figure()

#1 Plot Logistic Regression ROC
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

#2 Plot Random Forest ROC
plt.plot(rf_fpr, rf_tpr, label='Random Forest (area = %0.2f)' % rf_roc_auc)

#3 Plot Decision Tree ROC
plt.plot(dt_fpr, dt_tpr, label='Decision Tree (area = %0.2f)' % dt_roc_auc)

#4 Plot AdaBoost ROC
plt.plot(ada_fpr, ada_tpr, label='Adative Boosting (area = %0.2f)' % ada_roc_auc)

#5 Plot SVM ROC
plt.plot(svc_fpr, svc_tpr, label='SVM (area = %0.2f)' % svc_roc_auc)

#6 Plot ANN ROC
plt.plot(clf_fpr,clf_tpr, label='ANN (area = %0.2f)' % clf_roc_auc)


# Plot Base Rate ROC
plt.plot([0,1], [0,1],label='Base Rate')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Graph')
plt.legend(loc="lower right")
plt.show()





