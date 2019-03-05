#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 10:20:58 2019

@author: vladgriguta
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pickle
import os
import itertools

#ML libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

"""
import warnings
warnings.filterwarnings("ignore")
"""

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pd.DataFrame(pickle.load(f))

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix',
                          cmap=plt.cm.Blues,directory=''):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Showing normalized confusion matrix")
    else:
        print('Showing confusion matrix, without normalization')
    #print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),fontsize = 6,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(directory+title+'.png')
    plt.gcf().clear()
    
    
def plot_feature_importance(data,pipeline,title,directory=''):
    #make plot of feature importances
    clf=pipeline.steps[0][1] #get classifier used. zero because only 1 step.
    importances = pipeline.steps[0][1].feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)
    indices = np.argsort(importances)[::-1]
    feature_names_importanceorder=[]
    for f in range(len(indices)):
        #print("%d. feature %d (%f) {0}" % (f + 1, indices[f], importances[indices[f]]), feature_names[indices[f]])
        feature_names_importanceorder.append(str(data['feature_names'][indices[f]]))
    plt.figure()
    plt.title(title)
    plt.bar(range(len(indices)), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(len(indices)), indices)
    plt.xlim([-1, len(indices)])
    plt.xticks(range(len(indices)), feature_names_importanceorder, rotation='vertical')
    plt.tight_layout()
    plt.savefig(directory+title+'.png',format='png')
    plt.show()
    #plt.gcf().clear()



def prepare_data_classes_NN(input_table, trim_columns,train_percent=0.7):
    
    data_table=load_obj(input_table)
    
    
    # Drop the entries that do not have a class assigned
    data_table = data_table.replace(np.nan, 'X', regex=True)
    data_table.drop(index=data_table[data_table['class']=='X'].index,inplace=True)
    
    data_table.reset_index(inplace=True)
    
    y=data_table['class']
    
    #trim away unwanted columns    
    x=data_table.drop(columns=trim_columns)

    name_of_features = x.columns
    
    # Scale all data
    scaler = MinMaxScaler()
    scaler.fit(x)
    x = scaler.transform(x)
    
    
    from keras.utils import np_utils
    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_Y = encoder.transform(y)
    
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = np_utils.to_categorical(encoded_Y)
    
    # compute weights to account for class imbalace and improve f1 score
    y_cat = np.unique(encoded_Y)
    class_appearences = {y_cat[i]:np.sum(encoded_Y==y_cat[i]) for i in range(len(y_cat))}
    n_classes_norm = len(encoded_Y)/10000
    class_weights = {list(class_appearences.keys())[i]:n_classes_norm/list(class_appearences.values())[i] for i in range(len(class_appearences))}
    
    
    #split data up into test/train
    x_train, x_test, dummy_y_train, dummy_y_test = train_test_split(x,
                    dummy_y, train_size=train_percent, random_state=0)
    
    
    return x_train, x_test, dummy_y_train, dummy_y_test,encoder,class_weights, name_of_features



















