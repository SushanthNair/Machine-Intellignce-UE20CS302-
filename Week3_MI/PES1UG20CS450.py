'''
Assume df is a pandas dataframe object of the dataset given

'''

import numpy as np
import pandas as pd
import random


'''Calculate the entropy of the enitre dataset'''
# input:pandas_dataframe
# output:int/float
def get_entropy_of_dataset(df):
    s = 0 #entropy
    target = df[[df.columns[-1]]].values
    _, cnt = np.unique(target, return_counts=True)
    total = np.sum(cnt)
    for i in cnt:
        temp = i/total
        if temp != 0:
            s -= temp*(np.log2(temp))
    return s


'''Return avg_info of the attribute provided as parameter'''
# input:pandas_dataframe,str   {i.e the column name ,ex: Temperature in the Play tennis dataset}
# output:int/float
def get_avg_info_of_attribute(df, attribute):
    # TODO
    attr_val = df[attribute].values
    unique_attr_val = np.unique(attr_val)
    rows = df.shape[0]
    entropy_attr = 0
    for current_value in unique_attr_val:
        df_slice = df[df[attribute] == current_value]
        target = df_slice[[df_slice.columns[-1]]].values
        _, cnts = np.unique(target, return_counts=True)
        total_count = np.sum(cnts)
        entropy = 0
        for i in cnts:
            temp = i/total_count
            if temp != 0:
                entropy -= temp*np.log2(temp)
        entropy_attr += entropy*(np.sum(cnts)/rows)
    return(abs(entropy_attr))

'''Return Information Gain of the attribute provided as parameter'''
# input:pandas_dataframe,str
# output:int/float
def get_information_gain(df, attribute):
    gain = 0
    entropy_attr = get_avg_info_of_attribute(df, attribute)
    entropy_of_dataset = get_entropy_of_dataset(df)
    gain = entropy_of_dataset - entropy_attr
    return gain





#input: pandas_dataframe
#output: ({dict},'str')
def get_selected_attribute(df):
    '''
    Return a tuple with the first element as a dictionary which has IG of all columns 
    and the second element as a string with the name of the column selected

    example : ({'A':0.123,'B':0.768,'C':1.23} , 'C')
    '''
    info_gains = {}
    column = ''

    max_info_gain = float("-inf")
    for i in df.columns[:-1]:
        information_gain_of_attribute = get_information_gain(df, i)
        if information_gain_of_attribute > max_info_gain:
            column = i
            max_info_gain = information_gain_of_attribute
        info_gains[i] = information_gain_of_attribute
    return (info_gains, column)
