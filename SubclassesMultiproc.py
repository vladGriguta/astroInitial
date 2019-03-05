#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 14:58:34 2019

@author: vladgriguta
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

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
    #data_table.drop(index=data_table[data_table['subclass']=='X'].index,inplace=True)
    
    data_table.reset_index(inplace=True)
    import warnings
    warnings.filterwarnings("ignore")
    # Create dummy labels for the class-subclass combination
    data_table['ClassAndSubclass'] = ''
    
    data_table['ClassAndSubclass'][(data_table['class']=='GALAXY') & (data_table['subclass']=='X')] = 'G N/A'
    data_table['ClassAndSubclass'][(data_table['class']=='GALAXY') & (data_table['subclass']=='STARFORMING')] = 'G STF'
    data_table['ClassAndSubclass'][(data_table['class']=='GALAXY') & (data_table['subclass']=='BROADLINE')] = 'G BRL'
    data_table['ClassAndSubclass'][(data_table['class']=='GALAXY') & (data_table['subclass']=='STARBURST')] = 'G STB'
    data_table['ClassAndSubclass'][(data_table['class']=='GALAXY') & (data_table['subclass']=='AGN')] = 'G AGN'
    data_table['ClassAndSubclass'][(data_table['class']=='GALAXY') & (data_table['subclass']=='AGN BROADLINE')] = 'G AGNB'    
    data_table['ClassAndSubclass'][(data_table['class']=='GALAXY') & (data_table['subclass']=='STARBURST BROADLINE')] = 'G STBR'
    data_table['ClassAndSubclass'][(data_table['class']=='GALAXY') & (data_table['subclass']=='STARFORMING BROADLINE')] = 'G STFBR'
    
    data_table['ClassAndSubclass'][(data_table['class']=='QSO') & (data_table['subclass']=='X')] = 'Q N/A'
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
    
    # compute weights to account for class imbalace and improve f1 score
    y_cat = np.unique(encoded_Y)
    class_appearences = {y_cat[i]:np.sum(encoded_Y==y_cat[i]) for i in range(len(y_cat))}
    n_classes_norm = len(encoded_Y)/10000
    class_weights = {list(class_appearences.keys())[i]:n_classes_norm/list(class_appearences.values())[i] for i in range(len(class_appearences))}
    
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = np_utils.to_categorical(encoded_Y)
    
    
    #split data up into test/train
    x_train, x_test, dummy_y_train, dummy_y_test = train_test_split(x,
                    dummy_y, train_size=train_percent, random_state=0)
    
    
    return x_train, x_test, dummy_y_train, dummy_y_test,encoder,class_weights, name_of_features
    
    
def NeuralNet(trim_columns,input_table='test_query_table_100k', n_jobs=-1,):
    
    
    x_train, x_test, dummy_y_train, dummy_y_test,encoder,class_weights,_ = prepare_data_NN(input_table, trim_columns)
    input_dim = len(x_train[0])
    output_dim = len(dummy_y_train[0])

    # Importing the Keras libraries and packages
    from keras.models import Sequential
    from keras.utils import Sequence
    from keras.layers import Dense
    from keras.layers import Dropout
    
    
    class DataSequenceGenerator(Sequence):
    
        def __init__(self, x_train, y_train, batch_size):
            self.x, self.y = x_train, y_train
            self.batch_size = batch_size
    
        def __len__(self):
            return int(np.ceil(len(self.x) / float(self.batch_size)))
    
        def __getitem__(self, idx):
            batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

            return np.array(batch_x), np.array(batch_y)
    
    
    params = {'batch_size': int(len(x_train)/256)}
    
    training_generator = DataSequenceGenerator(x_train, dummy_y_train, **params)
    
    
    
    # define baseline model
    def baseline_model():
       	# create model
        model = Sequential()
        model.add(Dense(units = 32, kernel_initializer = 'VarianceScaling', activation = 'relu', input_dim=input_dim))
        #model.add(Dropout(0.3))
        model.add(Dense(units = 16, kernel_initializer = 'VarianceScaling', activation = 'sigmoid'))
        #model.add(Dropout(0.3))
        model.add(Dense(units = 14, kernel_initializer = 'VarianceScaling', activation = 'sigmoid'))
        #model.add(Dropout(0.3))
        model.add(Dense(units=output_dim, activation='softmax'))
    	# Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
        
        return model
    

    classifier = baseline_model()
    
    
    #classifier.fit(x_train, dummy_y_train,class_weight=class_weights, batch_size = 16, epochs = 10,verbose=True)
    classifier.fit_generator(generator=training_generator,class_weight=class_weights,epochs=300,use_multiprocessing=True,workers=-1)
    
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
    f1 = f1_score(encoded_y_test,predictions,average='macro')        
    # Compute and plot the confusion matrix
    #cnf_matrix = confusion_matrix(encoded_y_test, predictions,labels=classes_y_test)
    acc = accuracy_score(encoded_y_test,predictions)
    
    evaluation = classifier.evaluate(x=x_train, y=dummy_y_train, batch_size=16)
    print(evaluation)
    
    print('The f1 score of NN is:   '+str(f1))
    print('The accuracy of NN is:   '+str(acc))
    
    return encoded_y_test, predictions, classes_y_test

if __name__ == "__main__":
    
    input_table = '../moreData/test_query_table_1M'
    trim_columns=['#ra', 'dec', 'z', 'peak','integr','rms','subclass','class']


    encoded_y_test, predictions, classes_y_test = NeuralNet(trim_columns)
    
    
    #cnf_matrix = confusion_matrix(encoded_y_test, predictions,labels=classes_y_test)
    
    
    
    
    
