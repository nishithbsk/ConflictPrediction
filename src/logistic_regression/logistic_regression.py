import numpy as np

def load_data(data_path='../../data/processed/logistic_regression/uganda.csv'):
    # returns tuple (X_train, X_test, y_train, y_test)
    return np.load(data_path)
