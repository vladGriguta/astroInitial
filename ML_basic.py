#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 14:19:06 2019

@author: vladgriguta
"""

import os, sys, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Need this line to display through the entire pipeline (external->personal device)
import matplotlib

import pickle
import time
import itertools
from textwrap import wrap
import multiprocessing
import time

#ML libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import scipy.stats as stats
from sklearn import manifold
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE #single core TSNE, sklearn.
# Need 'pip install MulticoreTSNE' before. 
from MulticoreTSNE import MulticoreTSNE as multiTSNE #multicore TSNE, not sklearn implementation.


###############################################################################

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pd.DataFrame(pickle.load(f))
    
#Function to prepare data for Machine Learning
def prepare_data(filename, trim_columns, train_percent=0.7, tsne=False):
    
    data_table=load_obj(filename)

    #trim away unwanted columns    
    all_features=data_table.drop(columns=trim_columns+['class'])
    all_classes=data_table['class']
    
    #split data up into test/train
    features_train, features_test, classes_train, classes_test = train_test_split(all_features,
                    all_classes, train_size=train_percent, random_state=0, stratify=all_classes)
    class_names=np.array(np.unique(all_classes))
    feature_names=list(all_features)

    #return dictionary: features_train, features_test, classes_train, classes_test, class_names, feature_names
    if tsne==False:
        return {'features_train':features_train, 'features_test':features_test,
                'classes_train':classes_train, 'classes_test':classes_test,
                'class_names':class_names, 'feature_names':feature_names}
    if tsne==True:
        return {'all_features':all_features, 'all_classes':all_classes}


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix',
                          cmap=plt.cm.Blues):
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
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('MLResults/ConfusionMatrix.png')
    plt.gcf().clear()
    
    
#Function to run randon forest pipeline with feature pruning and analysis
def RF_pipeline(data, train_percent, n_jobs=-1, n_estimators=500):
    #rfc=RandomForestClassifier(n_jobs=n_jobs,n_estimators=n_estimators,random_state=2,class_weight='balanced')
    pipeline = Pipeline([ ('classification', RandomForestClassifier(n_jobs=n_jobs,
            n_estimators=n_estimators,random_state=0,class_weight='balanced')) ])
    #do the fit and feature selection
    pipeline.fit(data['features_train'], data['classes_train'])
    # check accuracy and other metrics:
    classes_pred = pipeline.predict(data['features_test'])
    accuracy_before=(accuracy_score(data['classes_test'], classes_pred))

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
    plt.title("\n".join(wrap("Feature importances. n_est={0}. Trained on {1}% of data."+
        " Accuracy before={2:.3f}".format(n_estimators,train_percent*100,accuracy_before))))
    plt.bar(range(len(indices)), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(len(indices)), indices)
    plt.xlim([-1, len(indices)])
    plt.xticks(range(len(indices)), feature_names_importanceorder, rotation='vertical')
    plt.tight_layout()
    plt.savefig('MLResults/Feature_importances.png')
    plt.gcf().clear()
    
    return classes_pred



if __name__ == "__main__":
    
    input_table = 'test_query_table_100k'
    trim_columns=['#ra', 'dec', 'z', 'peak','integr','rms','subclass']

    data = prepare_data(input_table,trim_columns,train_percent=0.7)
    
    # Call the Random Forest classifier to do the job
    classes_pred = RF_pipeline(data,train_percent=0.7,n_jobs=3,n_estimators=500)
    
    # Compute and plot the confusion matrix
    cnf_matrix = confusion_matrix(data['classes_test'], classes_pred)
    plot_confusion_matrix(cnf_matrix, classes=data['class_names'],
                          title='Confusion matrix, without normalization')
    
    # Compute the F1 Score
    labels = data['class_names']
    f1 = f1_score(data['classes_test'],classes_pred,labels=labels,average='weighted')
    
    
    
    
    
    
    
    
    
    
    