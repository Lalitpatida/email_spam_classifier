#data collection
#data preprocessing
#data visualization
#model training 
#model testing
#model evaluation
#model deployment
#model monitoring 
#model update 

import warnings
warnings.simplefilter('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data\dataset.csv")
print(df.head(10))
print(df.isnull().sum())

# #data preprocessing 
df['Category'] = df['Category'].map({'spam': 0, 'ham': 1})
print(df)

X = df['Message']
Y = df['Category']

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=3)

#feature engineering 
from sklearn.feature_extraction.text import TfidfVectorizer

tf = TfidfVectorizer(min_df=1,stop_words='english',lowercase=True)
X_train_features = tf.fit_transform(X_train)
X_test_features = tf.transform(X_test)


# model training and testing

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC

#logistic regression
lr = LogisticRegression()
lr.fit(X_train_features,Y_train)
lr_train = lr.predict(X_train_features)
lr_test = lr.predict(X_test_features)

#decision tree 
dtree = DecisionTreeClassifier()
dtree.fit(X_train_features,Y_train)
dtree_train = dtree.predict(X_train_features)
dtree_test = dtree.predict(X_test_features)

#K nearest neighbour
knn = KNeighborsClassifier()
knn.fit(X_train_features,Y_train)
knn_train = knn.predict(X_train_features)
knn_test = knn.predict(X_test_features)

# random forest
rn = RandomForestClassifier()
rn.fit(X_train_features,Y_train)
rn_train = rn.predict(X_train_features)
rn_test = rn.predict(X_test_features)

# ensemble stacking
estimators = [('lr',lr),('dtree',dtree),('knn',knn),('rf',rn)]
stack = StackingClassifier(estimators,final_estimator = SVC(kernel='linear'))
stack.fit(X_train_features,Y_train)
stack_train = stack.predict(X_train_features)
stack_test = stack.predict(X_test_features)


# model evaluation
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

#logistic regression
lr_train_acc = accuracy_score(Y_train,lr_train)
lr_test_acc = accuracy_score(Y_test,lr_test)
lr_precision = precision_score(Y_test,lr_test)
lr_recall = recall_score(Y_test,lr_test)
lr_f1 = f1_score(Y_test,lr_test)

#decision tree
dt_train_acc = accuracy_score(Y_train,dtree_train)
dt_test_acc = accuracy_score(Y_test, dtree_test)
dt_precision = precision_score(Y_test,dtree_test)
dt_recall = recall_score(Y_test,dtree_test)
dt_f1 = f1_score(Y_test,dtree_test)

# K nearest neighbour
knn_train_acc = accuracy_score(Y_train,knn_train)
knn_test_acc = accuracy_score(Y_test,knn_test)
knn_precision = precision_score(Y_test,knn_test)
knn_recall = recall_score(Y_test,knn_test)
knn_f1 = f1_score(Y_test,knn_test)

#random forest 
rf_train_acc = accuracy_score(Y_train,rn_train)
rf_test_acc = accuracy_score(Y_test,rn_test)
rf_precision = precision_score(Y_test,rn_test)
rf_recall = recall_score(Y_test,rn_test)
rf_f1 = f1_score(Y_test,rn_test)

# stacking
stack_train_acc = accuracy_score(Y_train,stack_train)
stack_test_acc = accuracy_score(Y_test,stack_test)
stack_precision = precision_score(Y_test,stack_test)
stack_recall = recall_score(Y_test,stack_test)
stack_f1 = f1_score(Y_test,stack_test)



import pandas as pd

results = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree', 'KNN', 'Random Forest', 'Stacking'],
    'Train Accuracy': [lr_train_acc, dt_train_acc, knn_train_acc, rf_train_acc, stack_train_acc],
    'Test Accuracy': [lr_test_acc, dt_test_acc, knn_test_acc, rf_test_acc, stack_test_acc],
    'Precision': [lr_precision, dt_precision, knn_precision, rf_precision, stack_precision],
    'Recall': [lr_recall, dt_recall, knn_recall, rf_recall, stack_recall],
    'F1 Score': [lr_f1, dt_f1, knn_f1, rf_f1, stack_f1]
})

print(results)

with open("results.txt", "w") as f:
    f.write("===== MODEL PERFORMANCE =====\n\n")
    f.write(results.to_string(index=False))


import os

os.makedirs("models", exist_ok=True)

import pickle

# Logistic Regression
with open('models/lr_model.pkl', 'wb') as f:
    pickle.dump(lr, f)

# Decision Tree
with open('models/dt_model.pkl', 'wb') as f:
    pickle.dump(dtree, f)

# KNN
with open('models/knn_model.pkl', 'wb') as f:
    pickle.dump(knn, f)

# Random Forest
with open('models/rf_model.pkl', 'wb') as f:
    pickle.dump(rn, f)

# Stacking Model
with open('models/stack_model.pkl', 'wb') as f:
    pickle.dump(stack, f)


with open('models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tf, f)

    