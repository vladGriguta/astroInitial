#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 10:45:49 2019

@author: vladgriguta
"""


#import os, sys, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pickle
import os
import time
import itertools
from textwrap import wrap
#import multiprocessing
#import time

#ML libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
#from sklearn.feature_selection import SelectFromModel
#from sklearn.metrics import classification_report

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
def prepare_data_subclass_galaxies(input_table, trim_columns, train_percent=0.7, tsne=False,additionalFeatures=False):
    
    data_table=load_obj(input_table)
    
    # drop stars
    data_table.drop(index=data_table[data_table['class']=='STAR'].index,inplace=True)
    
    # Drop the entries that do not have a subclass assigned
    data_table = data_table.replace(np.nan, 'X', regex=True)
    data_table.drop(index=data_table[data_table['subclass']=='X'].index,inplace=True)
    
    data_table['subclass'].unique()
    
    # Create dummy labels for the class-subclass combination
    data_table['ClassAndSubclass'] = ''
    data_table['ClassAndSubclass'][(data_table['class']=='GALAXY') & (data_table['subclass']=='STARFORMING')] = 'G STF'
    data_table['ClassAndSubclass'][(data_table['class']=='GALAXY') & (data_table['subclass']=='BROADLINE')] = 'G BRL'
    data_table['ClassAndSubclass'][(data_table['class']=='GALAXY') & (data_table['subclass']=='STARBURST')] = 'G STB'
    data_table['ClassAndSubclass'][(data_table['class']=='GALAXY') & (data_table['subclass']=='AGN')] = 'G AGN'
    data_table['ClassAndSubclass'][(data_table['class']=='GALAXY') & (data_table['subclass']=='AGN BROADLINE')] = 'G AGNB'    
    data_table['ClassAndSubclass'][(data_table['class']=='GALAXY') & (data_table['subclass']=='STARBURST BROADLINE')] = 'G STBR'
    data_table['ClassAndSubclass'][(data_table['class']=='GALAXY') & (data_table['subclass']=='STARFORMING BROADLINE')] = 'G STFBR'

    data_table['ClassAndSubclass'][(data_table['class']=='QSO') & (data_table['subclass']=='STARFORMING')] = 'Q STF'
    data_table['ClassAndSubclass'][(data_table['class']=='QSO') & (data_table['subclass']=='BROADLINE')] = 'Q BRL'
    data_table['ClassAndSubclass'][(data_table['class']=='QSO') & (data_table['subclass']=='STARBURST')] = 'Q STB'  
    data_table['ClassAndSubclass'][(data_table['class']=='QSO') & (data_table['subclass']=='AGN')] = 'Q AGN'  
    data_table['ClassAndSubclass'][(data_table['class']=='QSO') & (data_table['subclass']=='AGN BROADLINE')] = 'Q AGNB'      
    data_table['ClassAndSubclass'][(data_table['class']=='QSO') & (data_table['subclass']=='STARBURST BROADLINE')] = 'Q STBR'   
    data_table['ClassAndSubclass'][(data_table['class']=='QSO') & (data_table['subclass']=='STARFORMING BROADLINE')] = 'Q STFBR'
    
    
    
    # save the classes to predict        
    #data_table['ClassAndSubclass'] = data_table['class'] +', '+ data_table['subclass']
    classes=data_table['ClassAndSubclass']

    #trim away unwanted columns    
    features=data_table.drop(columns=trim_columns+['ClassAndSubclass'])
    if(additionalFeatures):
        features['w1-w3'] = features['w1'] - features['w3']
        features['mag_z-mag_u'] = features['mag_z'] - features['mag_u']

    name_of_features = features.columns
    
    # Scale all data
    scaler = MinMaxScaler()
    scaler.fit(features)
    features = scaler.transform(features)
    
    #split data up into test/train
    features_train, features_test, classes_train, classes_test = train_test_split(features,
                    classes, train_size=train_percent, random_state=0, stratify=classes)
    class_names=np.array(np.unique(classes))

    #return dictionary: features_train, features_test, classes_train, classes_test, class_names, feature_names
    if tsne==False:
        return {'features_train':features_train, 'features_test':features_test,
                'classes_train':classes_train, 'classes_test':classes_test,
                'class_names':class_names, 'feature_names':name_of_features}, scaler
    if tsne==True:
        return {'all_features':features, 'all_classes':classes}, scaler


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
        plt.text(j, i, format(cm[i, j], fmt),
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
    plt.gcf().clear()
    
#Function to run randon forest pipeline with feature pruning and analysis
def RF_pipeline(data, train_percent, n_jobs=-1, n_estimators=500,directory=''):
    #rfc=RandomForestClassifier(n_jobs=n_jobs,n_estimators=n_estimators,random_state=2,class_weight='balanced')
    pipeline = Pipeline([ ('classification', RandomForestClassifier(n_jobs=n_jobs,
            n_estimators=n_estimators,random_state=0,class_weight='balanced')) ])
    #do the fit and feature selection
    pipeline.fit(data['features_train'], data['classes_train'])
    # check accuracy and other metrics:
    classes_pred = pipeline.predict(data['features_test'])
    accuracy=(accuracy_score(data['classes_test'], classes_pred))
    # Compute the F1 Score
    f1 = f1_score(data['classes_test'],classes_pred,labels=data['class_names'],average='weighted')    

    #make plot of feature importances
    plot_feature_importance(data,pipeline,title='Feature Importance RF',directory=directory)
    
    # Compute and plot the confusion matrix
    cnf_matrix = confusion_matrix(data['classes_test'], classes_pred)
    plot_confusion_matrix(cnf_matrix, classes=data['class_names'],
                          title='Confusion matrix Random Forest',directory=directory)
    
    return classes_pred,accuracy,f1
    

def linear_classifier(data,train_percent,n_jobs=-1,additionalFeatures=False,directory=''):
    pipeline = Pipeline([('classification', LogisticRegression(n_jobs=3,solver='newton-cg',
                tol=1e-5,max_iter=500,class_weight='balanced',multi_class='auto'))])
    pipeline.fit(data['features_train'],data['classes_train'])
    # check accuracy and other metrics:
    classes_pred = pipeline.predict(data['features_test'])
    accuracy=(accuracy_score(data['classes_test'], classes_pred))
    f1 = f1_score(data['classes_test'],classes_pred,average='weighted')
    
    #plot_feature_importance(data,pipeline,title='Feature Importance LogisticReg')
    
    # Compute and plot the confusion matrix
    cnf_matrix = confusion_matrix(data['classes_test'], classes_pred)
    if(additionalFeatures):
        plot_confusion_matrix(cnf_matrix, classes=data['class_names'],
                          title='Confusion matrix Linear +features',directory=directory)
    else:
        plot_confusion_matrix(cnf_matrix, classes=data['class_names'],
                          title='Confusion matrix Linear',directory=directory)
    
    # Plot feature importance
    print('Now plotting the feature importances...\n')

    feat_importances = pd.Series(np.abs(np.array(pipeline.steps[0][1].coef_[0])), index=data['feature_names'])
    feat_importances.nlargest(10).plot(kind='barh')
    plt.yticks(rotation=45)
    print('Finished Feature importance plotting.')
    if(additionalFeatures):
        plt.savefig(directory+'LinearFeatureImportanceAdditionalFeatures.png')
        plt.gcf().clear()
    else:
        plt.savefig(directory+'LinearFeatureImportance.png')
        plt.gcf().clear()
    
    return classes_pred,accuracy,f1
    
def SVC_classifier(data,train_percent,additionalFeatures=False,directory=''):
    pipeline = Pipeline([('classification', LinearSVC(loss='hinge', tol=1e-5,
                    max_iter=500,class_weight='balanced',multi_class='ovr'))])
    pipeline.fit(data['features_train'],data['classes_train'])
    # check accuracy and other metrics:
    classes_pred = pipeline.predict(data['features_test'])
    accuracy=(accuracy_score(data['classes_test'], classes_pred))
    f1 = f1_score(data['classes_test'],classes_pred,labels=data['class_names'],average='weighted')
    
    #plot_feature_importance(data,pipeline,title='Feature Importance LogisticReg')
    
    # Compute and plot the confusion matrix
    cnf_matrix = confusion_matrix(data['classes_test'], classes_pred)
    if(additionalFeatures):
        plot_confusion_matrix(cnf_matrix, classes=data['class_names'],
                          title='Confusion matrix SVC +features',directory=directory)
    else:
        plot_confusion_matrix(cnf_matrix, classes=data['class_names'],
                          title='Confusion matrix SVC',directory=directory)
    
    # Plot feature importance
    print('Now plotting the feature importances...\n')

    feat_importances = pd.Series(np.abs(np.array(pipeline.steps[0][1].coef_[0])), index=data['feature_names'])
    feat_importances.nlargest(10).plot(kind='barh')
    plt.yticks(rotation=45)
    print('Finished Feature importance plotting.')
    if(additionalFeatures):
        plt.savefig(directory+'SVCFeatureImportanceAdditionalFeatures.png')
    else:
        plt.savefig(directory+'SVCFeatureImportance.png')
    
    return classes_pred,accuracy,f1

if __name__ == "__main__":
    
    input_table = 'test_query_table_100k'
    trim_columns=['#ra', 'dec', 'z', 'peak','integr','rms','subclass','class']

    
    data, scaler = prepare_data_subclass_galaxies(input_table,trim_columns,train_percent=0.7,additionalFeatures=False)
    
    directory = 'MLsubclass_2802/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    classes_pred,accuracy_LR,f1_LR = linear_classifier(data,train_percent=0.7,n_jobs=-1,
                                            additionalFeatures=False,directory=directory)
    print('The accuracy of LR is a = '+str(accuracy_LR))
    print('The f1 score of LR is a = '+str(f1_LR))
    
    
    classes_pred,accuracy_SVC,f1_SVC = SVC_classifier(data,train_percent=0.7,
                                            additionalFeatures=False,directory=directory)
    print('The accuracy of SVC is a = '+str(accuracy_SVC))
    print('The f1 score of SVC is a = '+str(f1_SVC))
    
    
    classes_pred_RF,accuracy_RF,f1_RF = RF_pipeline(data, train_percent=0.7, n_jobs=-1,
                                                    n_estimators=500,directory=directory)
    print('The accuracy of RF is a = '+str(accuracy_RF))
    print('The f1 score of RF is a = '+str(f1_RF))
    
    """
    # Check if the accuracy of 80% is true
    comparisonList = np.array(np.stack((np.array(data['classes_test']),np.array(classes_pred_RF)),axis=1))
    erroneousClassification = 0
    for i in range(len(comparisonList)):
        if(comparisonList[i][0] != comparisonList[i][1]):
            erroneousClassification += 1
    print('The accuracy of RF is: '+str(100*(1-(erroneousClassification/len(comparisonList))))+' %')
    """
    