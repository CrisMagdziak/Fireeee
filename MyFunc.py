import numpy as np
import pandas as pd

def classify_row(value):
    """ Function that check if or contains word Dwelling

    Args:
        value -> value from column

    Return: returns string Business or Non-Business
    
    """
    if 'Dwelling' in value:
        return 'Business '
    else:
        return 'Non-Business'
    

def dataset_IQR(col) :
    """ Calculate IQR for given column
     
    Args:
        col -> column for IQR calculation

    Return:
        IQR value of specific column, value -> float
    
    """
    return np.quantile(col, 0.75) - np.quantile(col, 0.25)

def lower_tresh(col):
    """ Calculate lower treshold

    Args:
        col -> column for IQR calculation

    Return:
        lower outliner value -> float
    """ 
    return np.quantile(col, 0.25) - (1.5 * dataset_IQR(col))

def upper_tresh(col) :
    """ Calculate upper treshold

    Args:
        col -> column for IQR calculation

    Return:
        upper outliner value -> float
    """ 
    return np.quantile(col, 0.75) + (1.5 * dataset_IQR(col))

def lower_treshholders(dataframe) :
    """ Check how many outliners-lower we have for each column
    
    Args:
        dataframe -> dataframe from where we load data
    
    """
    for x in dataframe.select_dtypes(include= 'number') :
        if dataframe[x][dataframe[x] < lower_tresh(dataframe[x])].count() > 0 :
            print(f'Column: {x}')
            print(f'Lower treshold {lower_tresh(dataframe[x])}')
            print(f'Ilosc wartosci ponizej dolnego outlinera: {dataframe[x][dataframe[x] < lower_tresh(dataframe[x])].count()}\n')
        

def upper_treshholders(dataframe) :
    """ Check how many outliners-upper we have for each column
    
    Args:
        dataframe -> dataframe from where we load data
    
    """
    for x in dataframe.select_dtypes(include= 'number') :
        if dataframe[x][dataframe[x] > upper_tresh(dataframe[x])].count() > 0 :
            print(f'Column: {x}')
            print(f'Upper treshold {upper_tresh(dataframe[x])}')
            print(f'Ilosc wartosci powyzej górnego outlinera: {dataframe[x][dataframe[x] > upper_tresh(dataframe[x])].count()}\n')


def calculate_woe(df, feature, target) :
    """ Function that calculate weight of evidence

        Args: 
            df -> dataframe
            target -> our target(y) feature
            feature -> column that we want to map

        Return: a map of the WOE values for the individual categories
    """ 
    df = df.copy()
    df['target'] = target
    categories = df[feature].unique()
    woe_map = {}

    for category in categories :
        total_good = df[df[feature] == category]['target'].sum()
        total_bad = df[df[feature] == category]['target'].count() - total_good
        woe = np.log((total_good / total_bad) / (df['target'].sum() / df['target'].count()))
        woe_map[category] = woe
    return woe_map


def is_numeric(col):
    """ Check if columns have only numeric values

        Args: 
            col -> columns where we are checking
    """
    try:
        pd.to_numeric(col)
        return True
    except ValueError:
        return False