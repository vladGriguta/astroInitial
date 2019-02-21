# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import numpy as np
from numpy import nan
import pandas as pd
import matplotlib.pyplot as plt
import pickle

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

def plot_distributions(data,directory):
    """
    Function that takes as input the database with magnitudes of objects for
    different photometric filters and outputs the distribution of magnitudes in
    photometric spectra for each subclass of the three classes ->(Galaxy,Quasars).
    The filters used in SDSS are:
        'mag_u' : 355.1,'mag_g' : 468.6,'mag_r' : 616.5,'mag_i' : 748.1,
        'mag_z' : 893.1, 'w1' : 3400,'w2' : 4600,'w3' : 12000,'w4' : 22000
    """
    
    # Create dictionary to translate from mag class to wavelength
    mag_to_wv = {'mag_u' : 355.1,'mag_g' : 468.6,'mag_r' : 616.5,'mag_i' : 748.1,
                 'mag_z' : 893.1, 'w1' : 3400,'w2' : 4600,'w3' : 12000,'w4' : 22000}
    wavelength = np.zeros(len(mag_to_wv))
    i = 0
    for element in data.columns:
        if(element == 'class' or element == 'subclass'):
            continue
        wavelength[i] = np.log10(mag_to_wv[element])
        i += 1
    # In this function we only look at galaxies (including quasars)
    data = data.drop(data[data['class']=='STAR'].index)
    
    
    data_G = data[data['class'] == 'GALAXY']
    data_Q = data[data['class'] == 'QSO']
    
    
    colors = ['darkred','r','orange','y','g','b','navy','violet','m']
    colors = colors[0:len(list(data['subclass'].unique()))]
    
    subclasses = list(data['subclass'].unique())
    
    # Start with quasars
    y = np.zeros(len(data_G.columns) - 2) # substract 'class' and 'subclass'
    for subclass in subclasses:
        i = 0
        for element in data_Q.columns:
            if(element == 'class' or element == 'subclass'):
                continue
            y[i] = np.mean(data_Q[data_Q['subclass']==subclass][element])
            i += 1
            
        plt.plot(wavelength,y,label = subclass,markersize=10)
        plt.scatter(wavelength,y,marker='o',s=10)
        plt.xlabel('Logarithmic Wavelength [log(nm)]')
        plt.ylabel('Mean Magnitude')
        plt.title('Photometric Spectra of Quasars')
        plt.legend(loc='lower left')
        plt.grid(True)
        plt.savefig(directory+'spectraQuasarsLog.png',format='png')
    plt.gcf().clear()
    
    # Now galaxies
    for subclass in subclasses:
        i = 0
        for element in data_G.columns:
            if(element == 'class' or element == 'subclass'):
                continue
            y[i] = np.mean(data_G[data_G['subclass']==subclass][element])
            i += 1
            
        plt.plot(wavelength,y,label = subclass,markersize=10)
        plt.scatter(wavelength,y,marker='o',s=10)
        plt.xlabel('Logarithmic Wavelength [log(nm)]')
        plt.ylabel('Mean Magnitude')
        plt.title('Photometric Spectra of Galaxies')
        plt.legend(loc='lower left')
        plt.grid(True)
        plt.savefig(directory+'spectraGalaxiesLog.png',format='png')
    plt.gcf().clear()






if __name__ == "__main__":
    
    trim_columns=['#ra', 'dec', 'z', 'peak','integr','rms']
    input_table = '../moreData/test_query_table_1M'
    data = prepare_data(input_table,trim_columns)
    data = data.replace(np.nan, 'Not Assigned', regex=True)
    
    directory = 'week4/SubclassSpectra/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    plot_distributions(data,directory)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    