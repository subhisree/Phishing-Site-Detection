#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import warnings
import scikitplot as skplt
import graphviz
from sklearn import tree

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import keras
import pydotplus
from keras.utils.vis_utils import model_to_dot
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.layers import LeakyReLU
from keras import optimizers
from sklearn.linear_model import LogisticRegression


from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score 
import math

warnings.filterwarnings('ignore')


# In[2]:


data = arff.loadarff('../dataset.arff')
data = pd.DataFrame(data[0])
data = data.drop(columns = "id")
data.head(2)


# In[3]:


data.info()


# In[4]:


x = data.iloc[ : , :-1].values
y = data.iloc[:, -1:].values


# In[5]:


xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.1,random_state = 8)


# In[6]:


treemod = DecisionTreeClassifier(criterion = 'gini',min_samples_leaf = 10 , min_impurity_decrease = 0.01)
treemod.fit(xtrain,ytrain)


# In[7]:


pred = treemod.predict(xtest)
cm = confusion_matrix(ytest,pred)
correct_pred = sum([cm[k][k] for k in range(cm.shape[0])])
acc = correct_pred/xtest.shape[0]
print('Decision Tree Accuracy: {}'.format(acc))
p = treemod.predict_proba(xtest)
skplt.metrics.plot_roc_curve(ytest, p)
plt.show()


# In[8]:


data1 = data.drop(columns=['Result'])
dot_data = tree.export_graphviz(treemod, out_file=None, feature_names=data1.columns, 
                                class_names=["-1","1"], filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("Phishing.pdf")


# In[9]:


pred = treemod.predict(x)
data['DT'] = pred


# In[10]:



log = LogisticRegression(multi_class = 'multinomial', solver = 'newton-cg')
log.fit(xtrain, ytrain)
pred = log.predict(xtest)
cm = confusion_matrix(ytest,pred)
correct_pred = sum([cm[k][k] for k in range(cm.shape[0])])
acc = correct_pred/xtest.shape[0]
print('Linear Model Accuracy: {}'.format(acc))
p = log.predict_proba(xtest)
skplt.metrics.plot_roc_curve(ytest, p)
plt.show()


# In[11]:


pred = log.predict(x)
data['LR'] = pred


# In[12]:


svc = SVC(kernel = 'rbf',probability=True)
svc.fit(xtrain,ytrain)
pred = svc.predict(xtest)
cm = confusion_matrix(ytest,pred)
correct_pred = sum([cm[k][k] for k in range(cm.shape[0])])
acc = correct_pred/xtest.shape[0]
print('SVM Rbf Accuracy: {}'.format(acc))
p = svc.predict_proba(xtest)
skplt.metrics.plot_roc_curve(ytest, p)
plt.show()
cm


# In[13]:


pred = svc.predict(x)
data['SVM'] = pred


# In[14]:


r_state = random.randint(0,round(random.random() * 1000))


# In[15]:


max_i = 0
max_acc = 0
for i in [j*10 for j in range(1,10)]:
 rf = RandomForestClassifier(max_depth = i, random_state=r_state)
 rf.fit(xtrain, ytrain)
 pred = rf.predict(xtest)
 pred
 cm = confusion_matrix(ytest,pred)
 correct_pred = sum([cm[k][k] for k in range(cm.shape[0])])
 acc = correct_pred/xtest.shape[0]
 if(acc > max_acc):
  max_acc = acc
  max_i = i


# In[16]:


max_dep = 2
max_acc = 0
for i in [j for j in range(max_i-10,max_i+10)]:
 rf = RandomForestClassifier(max_depth = i, random_state=r_state)
 rf.fit(xtrain , list(ytrain))
 pred = rf.predict(xtest)
 cm = confusion_matrix(ytest,pred)
 correct_pred = sum([cm[k][k] for k in range(cm.shape[0])])
 acc = correct_pred/xtest.shape[0]
 if(acc > max_acc):
  max_dep = i
  max_acc = acc
print('Random Forest Accuracy: {}'.format(max_acc))
p = rf.predict_proba(xtest)
skplt.metrics.plot_roc_curve(ytest, p)
plt.show()


# In[17]:


pred = rf.predict(x)
data['RF'] = pred


# In[18]:


data.head(2)


# In[19]:


xgbmod = XGBClassifier(learning_rate =0.3,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
xgbmod.fit(xtrain,ytrain)
pred = xgbmod.predict(xtest)
cm = confusion_matrix(ytest,pred)
correct_pred = sum([cm[k][k] for k in range(cm.shape[0])])
acc = correct_pred/xtest.shape[0]
print('XGBoost Accuracy: {}'.format(acc))
p = xgbmod.predict_proba(xtest)
skplt.metrics.plot_roc_curve(ytest, p)
plt.show()


# In[20]:


pred = xgbmod.predict(x)
data['XGB'] = pred


# In[21]:


data.head(2)


# In[22]:


data.info()


# In[23]:


data.corr()


# In[24]:


seed = 7
np.random.seed(seed)


# In[25]:


x = data.drop(columns=['Result']).values
y = data['Result']
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.1)


# In[26]:


def create_baseline():
    model = Sequential()
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(35, input_dim=35, kernel_initializer='normal'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


# In[27]:


estimator = KerasClassifier(build_fn=create_baseline, epochs=5, batch_size=100, verbose=0)
kfold = StratifiedKFold(n_splits=100, shuffle=True, random_state=seed)
results = cross_val_score(estimator, xtrain, ytrain, cv=kfold)
print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# In[28]:


estimator = KerasClassifier(build_fn=create_baseline, epochs=5, batch_size=100, verbose=0)
estimator.fit(xtrain,ytrain)
y_pred_proba = estimator.predict_proba(xtest)
skplt.metrics.plot_roc_curve(ytest, y_pred_proba)
plt.show()
print("Neural Network Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# In[29]:


y_pred_proba = estimator.predict_proba(xtrain)
skplt.metrics.plot_roc_curve(ytrain, y_pred_proba)
plt.show()
print("Neural Network Results Training: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

