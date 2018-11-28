import time

#for feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

# for confusion matrix
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn

#for diagrams
import matplotlib.pyplot as plt


def get_preds(model, X_train, X_test):
    """
    Runs the KNN algorithm and returns predictions
    """
    print("Fitting data...")
    start_time = time.time()
    model.fit(X_train)
    print(time.time() - start_time, "seconds to fit data")
    
    print("Making predictions...")
    start_time = time.time()
    predictions = model.predict(X_test)
    print(time.time() - start_time, "seconds to predict the classes")
    
    return predictions
    
def get_preds_fitpred(model, X_train, X_test):
    """
    Runs the KNN algorithm and returns predictions
    """
    print("Fitting data...")
    start_time = time.time()
    model.fit(X_train)
    print(time.time() - start_time, "seconds to fit data")
    
    print("Making predictions...")
    start_time = time.time()
    predictions = model.fit_predict(X_train)
    print(time.time() - start_time, "seconds to predict the classes")
    
    return predictions

def get_k_best(X, y, k):
    """
    returns the inidices of the k best attributes for each emotion type
    """
    kbest = SelectKBest(f_regression, k)
    kbest.fit(X, y)
    best_attributes = kbest.get_support(True)
    return best_attributes


def reduce_attr(data, attr_indices):
    """
    saves the data at certain indices (which are in attr_indices)
    into the new dataset
    """
    new = []
    for i in range(len(data)):
        new.append([data[i][index] for index in attr_indices])
    return new


def reduce_attr_emo(new, data, attr_indices):
    """
    saves the data at certain indices (which are in attr_indices) into new
    """
    new.append([data[index] for index in attr_indices])


def reduce_data_emo(top_attrs, x, y, num):
    """
    transforms the given dataset to keep num amount of attributes
    """
    new_data = []
    for i in range(len(x)):
        if (y[i] == 0):
            reduce_attr_emo(new_data, x[i], top_attrs[num][0]) # num best attributes for angry emotion
        if (y[i] == 1):
            reduce_attr_emo(new_data, x[i], top_attrs[num][1]) # num best attributes for disgust emotion
        if (y[i] == 2):
            reduce_attr_emo(new_data, x[i], top_attrs[num][2]) # num best attributes for fear emotion
        if (y[i] == 3):
            reduce_attr_emo(new_data, x[i], top_attrs[num][3]) # num best attributes for happy emotion
        if (y[i] == 4):
            reduce_attr_emo(new_data, x[i], top_attrs[num][4]) # num best attributes for sad emotion
        if (y[i] == 5):
            reduce_attr_emo(new_data, x[i], top_attrs[num][5]) # num best attributes for surprise emotion
        if (y[i] == 6):
            reduce_attr_emo(new_data, x[i], top_attrs[num][6]) # num best attributes for neutral emotion
    return new_data