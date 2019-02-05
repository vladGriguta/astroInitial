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
    
    return data_table_trim

def plot_histogram(x,n_bins,name,location):
    n,bins,patches = plt.hist(x,n_bins,density = True, faceColor = 'b',alpha = 0.75)
    plt.xlabel('Magnitude')
    plt.ylabel('Probability')
    plt.title(name)
    plt.grid(True)
    plt.savefig(location+name+'.png',format='png')
    plt.gcf().clear()

if __name__ == "__main__":
    input_table = 'test_query_table_top10k'
    trim_columns=['#ra', 'dec', 'z', 'class']

    data = prepare_data(input_table,trim_columns)
    
    # Plot histograms for each distribution in magnitude.
    location = 'histograms/'
    for element in data.columns:
        Magnitude = np.array(data[data[element]>0][element])
        name = element
        plot_histogram(Magnitude,100,name,location)
    
    
    
    



