# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# import libraries
import numpy as np
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
    # Generate len(data.columns) colors
    colors = ['darkred','r','orange','y','g','b','navy','violet','m']
    colors = colors[0:len(data.columns)]
    # Do the stuff
    i = 0
    for element in data.columns:
        if(element == 'class'):
            continue
        Magnitude = np.array(data[data[element]>0][element])
        plt.hist(Magnitude,n_bins,density = True,faceColor = colors[i],alpha = 0.4,label=element)
        i += 1
    plt.xlabel('Magnitude')
    plt.ylabel('Probability')
    plt.legend()
    plt.show()
    plt.savefig(location+'allDistributionsOverlapped.png',format='png')
    
def plot_distributions(data,location):
    """
    Function that takes as input the database with magnitudes of objects for
    different photometric filters and outputs the distribution of magnitude
    for each class of objects.
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
        x_GALAXY[i] = mag_to_wv[element]
        y_GALAXY[i] = np.mean(data_GALAXY[[element]])
        
        x_QSO[i] = mag_to_wv[element]
        y_QSO[i] = np.mean(data_QSO[[element]])
        
        x_STAR[i] = mag_to_wv[element]
        y_STAR[i] = np.mean(data_STAR[[element]])
        i += 1
        
    plt.scatter(x_GALAXY,y_GALAXY,label = 'Galaxy')
    plt.scatter(x_QSO,y_QSO,label = 'Quasar')
    plt.scatter(x_STAR,y_STAR,label = 'Star')
    
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Mean Magnitude')
    plt.title('Photometric Spectra')
    plt.legend()
    plt.grid(True)
    plt.savefig(location+'spectraCombined.png',format='png')
    

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

    
    """
    location = 'wavelengthDistributions/'
    plot_distributions(data,location)
    """
    location = 'histograms/'
    plot_histogram_combined(data,location,n_bins=100)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    