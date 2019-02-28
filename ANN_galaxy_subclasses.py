#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 13:59:10 2019

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline



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



def prepare_data_NN(input_table, trim_columns,train_percent=0.7):
    
    data_table=load_obj(input_table)
    
    # drop stars
    data_table.drop(index=data_table[data_table['class']=='STAR'].index,inplace=True)
    
    # Drop the entries that do not have a subclass assigned
    data_table = data_table.replace(np.nan, 'X', regex=True)
    data_table.drop(index=data_table[data_table['subclass']=='X'].index,inplace=True)
    
    data_table.reset_index(inplace=True)
    import warnings
    warnings.filterwarnings("ignore")
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
    
    y=data_table['ClassAndSubclass']

    #trim away unwanted columns    
    x=data_table.drop(columns=trim_columns+['ClassAndSubclass'])

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
    
    #split data up into test/train
    x_train, x_test, dummy_y_train, dummy_y_test = train_test_split(x,
                    dummy_y, train_size=train_percent, random_state=0)
    
    
    return x_train, x_test, dummy_y_train, dummy_y_test,encoder, name_of_features
    
    
def NeuralNet(trim_columns,input_table='test_query_table_100k', n_jobs=-1,):
    
    
    x_train, x_test, dummy_y_train, dummy_y_test,encoder,_ = prepare_data_NN(input_table, trim_columns)
    input_dim = len(x_train[0])
    output_dim = len(dummy_y_train[0])
    # define baseline model
    def baseline_model():
        # Importing the Keras libraries and packages
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.layers import Dropout
        	# create model
        model = Sequential()
        model.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu', input_dim=input_dim))
        model.add(Dropout(0.2))
        model.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu'))
        model.add(Dropout(0.2))
        model.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'sigmoid'))
        model.add(Dense(units=output_dim, activation='softmax'))
    	# Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        return model
    

    classifier = baseline_model()
    
    classifier.fit(x_train, dummy_y_train, batch_size = 16, epochs = 10,verbose=True)
    
    """
    estimator = KerasClassifier(build_fn=baseline_model, epochs=2, batch_size=1, verbose=0)
    
    kfold = KFold(n_splits=10, shuffle=True, random_state=0)
    
    results = cross_val_score(estimator, x, dummy_y, cv=kfold)
    
    print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
    """
    dummy_y_pred = classifier.predict(x_test)
    
    predictions = np.argmax(dummy_y_pred,axis=1)
    
    # from dummy back to class names
    encoded_y_test = np.zeros(len(dummy_y_test))
    for i in range(len(dummy_y_test)):
        encoded_y_test[i] = int(np.argmax(dummy_y_test[i]))
    classes_y_test = encoder.inverse_transform(encoded_y_test.astype(int))
    
    # Compute the F1 Score
    f1 = f1_score(encoded_y_test,predictions,average='weighted')        
    # Compute and plot the confusion matrix
    #cnf_matrix = confusion_matrix(encoded_y_test, predictions,labels=classes_y_test)
    acc = accuracy_score(encoded_y_test,predictions)
    
    print('The f1 score of NN is:   '+str(f1))
    print('The accuracy of NN is:   '+str(acc))
    
    return encoded_y_test, predictions, classes_y_test

if __name__ == "__main__":
    
    input_table = '../moreData/test_query_table_1M'
    trim_columns=['#ra', 'dec', 'z', 'peak','integr','rms','subclass','class']


    encoded_y_test, predictions, classes_y_test = NeuralNet(trim_columns)
