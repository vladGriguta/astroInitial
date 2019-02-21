# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# import libraries
import os
import numpy as np
from numpy import nan
import pandas as pd
import matplotlib.pyplot as plt

import pickle

# ML libraries
from sklearn.model_selection import train_test_split

# define functions

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pd.DataFrame(pickle.load(f))

#Function to prepare data for Machine Learning
def prepare_data(filename, trim_columns, verbose=True):
    if verbose==True: print('loading saved tables from disk: '+filename)
    data_table=load_obj(filename)
    if verbose==True: print('The table loaded is of shape: {0}'.format(data_table.shape))
    #trim away unwanted columns
    data_table_trim=data_table.drop(columns=trim_columns)
    
    return pd.DataFrame(data_table_trim)

def plot_histogram(x,n_bins,name,location):
    n,bins,patches = plt.hist(x,n_bins,density = True, faceColor = 'b',alpha = 0.75)
    plt.xlabel('Magnitude')
    plt.ylabel('Probability')
    plt.title(name)
    plt.grid(True)
    plt.savefig(location+name+'.png',format='png')
    plt.gcf().clear()
    
    
def plot_histogram_combined(data,location,n_bins=100):
    """
    Function that plots all histograms of the magnitudes on one figure.
    """
    
    # Generate len(data.columns) colors
    colors = ['darkred','r','orange','y','g','b','navy','violet','m']
    colors = colors[0:len(data.columns)]
    # Do the stuff
    i = 0
    for element in data.columns:
        if(element == 'class'):
            continue
        Magnitude = np.array(data[data[element]>0][element])
        plt.hist(Magnitude,n_bins,density = True,faceColor = colors[i],
                 alpha = 0.4,label=element)
        i += 1
    plt.xlabel('Magnitude')
    plt.ylabel('Probability')
    plt.grid(True)
    plt.legend()
    plt.show()
    plt.savefig(location+'allDistributionsOverlapped.png',format='png')
    
    
def plot_histogram_byObjects(data,location,n_bins=100):
    """
    Function that takes as input the database with magnitudes of objects for
    different photometric filters and outputs the distribution of magnitude
    for each class of objects.
    
    """

    data_G = data[data['class'] == 'GALAXY']
    data_Q = data[data['class'] == 'QSO']
    data_S = data[data['class'] == 'STAR']
    

    for element in data.columns:
        # skip over class
        if(element == 'class'):
            continue
        Magnitude_G = np.array(data_G[(data_G[element]>0) & (data_G[element]<100)][element])
        plt.hist(Magnitude_G,n_bins,density = True,faceColor = colors[0],
                 alpha = 0.4,label='Galaxy')
        Magnitude_Q = np.array(data_Q[(data_Q[element]>0) & (data_Q[element]<100)][element])
        plt.hist(Magnitude_Q,n_bins,density = True,faceColor = colors[1],
                 alpha = 0.4,label='Quasar')
        Magnitude_S = np.array(data_S[(data_S[element]>0) & (data_S[element]<100)][element])
        plt.hist(Magnitude_S,n_bins,density = True,faceColor = colors[2],
                 alpha = 0.4,label='Star')
        plt.xlabel('Magnitude')
        plt.ylabel('Probability')
        plt.title(element)
        plt.xlim(7,27.5)
        plt.grid(True)
        plt.legend()
        plt.savefig(location+element+'.png',format='png')
        plt.gcf().clear()
        
def plot_histogram_bySubclasses(data,directory,n_bins=30):
    """
    Function that takes as input the database with magnitudes of objects for
    different photometric filters and outputs the distribution of magnitude
    for each subclass of objects
    
    """
    data_G = data[data['class'] == 'GALAXY']
    data_Q = data[data['class'] == 'QSO']

    colors = ['darkred','r','orange','y','g','b','navy','violet','m']
    colors = colors[0:len(list(data['subclass'].unique()))]
    
    
    data_list = [data_Q,data_G]
    r = 0
    for data in data_list:    
        for element in data.columns:
            # skip over class
            if(element == 'class' or element == 'subclass'):
                continue
            Magnitude = data[(data[element]>0) & (data[element]<100)]
            subclasses = list(data['subclass'].unique())
            for i in range(len(subclasses)):
                plt.hist(Magnitude[Magnitude['subclass']==subclasses[i]][element],
                         n_bins,density = True,faceColor = colors[i], alpha = 0.4,
                         label=subclasses[i])
                #plt.xlim(7,27.5)
            plt.xlabel('Magnitude')
            plt.ylabel('Probability')
            plt.grid(True)
            plt.legend()
            if(r == 0):
                plt.title(element+' Quasar')
                plt.savefig(directory+element+'_Quasar.png',format='png')
            else:
                plt.title(element+' Galaxy')
                plt.savefig(directory+element+'_Galaxy.png',format='png')
            plt.gcf().clear()
        r += 1
    
def plot_distributions(data,location):
    """
    Function that takes as input the database with magnitudes of objects for
    different photometric filters and outputs the distribution of magnitudes in
    photometric spectra for the three classes ->(Galaxy,Stars,Quasars).
    The filters used in SDSS are:
        'mag_u' : 355.1,'mag_g' : 468.6,'mag_r' : 616.5,'mag_i' : 748.1,
        'mag_z' : 893.1, 'w1' : 3400,'w2' : 4600,'w3' : 12000,'w4' : 22000
    """
    
    # Create dictionary to translate from mag class to wavelength
    mag_to_wv = {'mag_u' : 355.1,'mag_g' : 468.6,'mag_r' : 616.5,'mag_i' : 748.1,
                 'mag_z' : 893.1, 'w1' : 3400,'w2' : 4600,'w3' : 12000,'w4' : 22000}
    data_GALAXY = data[data['class']=='GALAXY']
    x_GALAXY = np.zeros(len(data_GALAXY.columns)-1)
    y_GALAXY = np.zeros(len(data_GALAXY.columns)-1)
    
    data_QSO = data[data['class']=='QSO']
    x_QSO = np.zeros(len(data_QSO.columns)-1)
    y_QSO = np.zeros(len(data_QSO.columns)-1)
    
    data_STAR = data[data['class']=='STAR']
    x_STAR = np.zeros(len(data_STAR.columns)-1)
    y_STAR = np.zeros(len(data_STAR.columns)-1)
    i = 0
    for element in data.columns:
        if(element == 'class'):
            continue
        print(element)
        x_GALAXY[i] = np.log10(mag_to_wv[element])
        y_GALAXY[i] = np.mean(data_GALAXY[[element]])
        
        x_QSO[i] = np.log10(mag_to_wv[element])
        y_QSO[i] = np.mean(data_QSO[[element]])
        
        x_STAR[i] = np.log10(mag_to_wv[element])
        y_STAR[i] = np.mean(data_STAR[[element]])
        i += 1
        
    plt.scatter(x_GALAXY,y_GALAXY,label = 'Galaxy')
    plt.scatter(x_QSO,y_QSO,label = 'Quasar')
    plt.scatter(x_STAR,y_STAR,label = 'Star')
    
    plt.xlabel('Logarithmic Wavelength [log(nm)]')
    plt.ylabel('Mean Magnitude')
    plt.title('Photometric Spectra')
    plt.legend()
    plt.grid(True)
    plt.savefig(location+'spectraCombinedLog.png',format='png')
    

def plot_subclass_appearances(data,directory):
    
    data_G = data[data['class'] == 'GALAXY']
    data_Q = data[data['class'] == 'QSO']
    data_S = data[data['class'] == 'STAR']
    data_list = [data_Q,data_G,data_S]
    
    for data in data_list:
        subclasses = sorted(list(data['subclass'].unique()))
        # Turn features that are discrete into cathegorical variables
        map_subclasses = {subclasses[i]:i for i in range(len(subclasses))}
        data['subclassCat'] = data['subclass'].map(map_subclasses)
        
        hist_subclass = {subclasses[i]:round(100*len(data.loc[data['subclass'] 
            == subclasses[i]])/len(data),2) for i in range(len(subclasses))}
        fig = plt.figure()
        plt.bar(hist_subclass.keys(), hist_subclass.values())
        fig.autofmt_xdate()
        plt.xlabel('Subclass')
        plt.ylabel('Appearances (%)')
        plt.title(data['class'].iloc[0])
        plt.grid(True)
        plt.savefig(directory+str(data['class'].iloc[0])+'.png',format='png')
        plt.gcf().clear()



if __name__ == "__main__":
    
    input_table = 'test_query_table_top10k'
    trim_columns=['#ra', 'dec', 'z', 'peak','integr','rms']

    data = prepare_data(input_table,trim_columns)
    
    """
    # Plot histograms for each distribution in magnitude.
    location = 'histograms/'
    for element in data.columns:
        Magnitude = np.array(data[data[element]>0][element])
        name = element
        plot_histogram(Magnitude,100,name,location)
    """

    
    
    location = 'wavelengthDistributions/'
    plot_distributions(data,location)
    
    """
    location = 'histograms/'
    plot_histogram_combined(data,location,n_bins=100)
    """
    """
    location = 'histogramsByClass/'
    plot_histogram_byObjects(data,location,n_bins=100)
    """
    
    """
    input_table2 = '../moreData/test_query_table_1M'
    trim_columns2 = trim_columns + ['subclass']
    data2 = prepare_data(input_table2,trim_columns2)

    location = 'histogramsByClass1M/'
    plot_histogram_byObjects(data2,location,n_bins=100)
    """
    
    """
    # Not yet ready
    directory = 'histogramsBySubclass1M/'
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    plot_histogram_bySubclasses(data2,directory,trim_columns2)
    """
    
    trim_columns=['#ra', 'dec', 'z', 'peak','integr','rms']
    input_table2 = '../moreData/test_query_table_1M'
    data2 = prepare_data(input_table2,trim_columns)
    data2 = data2.replace(np.nan, 'Not Assigned', regex=True)
    
    """
    directory = 'SubclassAppearances/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    plot_subclass_appearances(data2,directory)
    """
    
    
    """
    directory = 'SubclassDistributions/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    plot_histogram_bySubclasses(data2,directory)
    """
    
    
    
    
    
    
    
    
    
    
    
    