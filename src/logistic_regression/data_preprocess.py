import os
import sys
import csv
import dateutil
import numpy as np
import matplotlib.pyplot as plt

from dateutil.parser import parse
from dateutil.relativedelta import relativedelta
from sklearn.cross_validation import train_test_split

csv_filename = '../../data/raw/uganda.csv'
country_name = 'uganda'
save_name = '../../data/processed/logistic_regression/uganda_data'
degree_interval = 0.5
num_timesteps = 4
num_features = 2

latitude_column = 19
longitude_column = 20
date_column = 3
death_column = 24
conflict_index = 0
death_index = 1

def get_train_test(X, y, test_size=0.10):
    return train_test_split(X, y, test_size=test_size, random_state=42)

def get_data(grid):
    X, y = [], []
    timescale, num_rows, num_cols, num_features = grid.shape
    for i in range(num_rows):
        for j in range(num_cols):
            grid_cell = grid[:, i, j, :]
            # ignore if cell has never had any conflicts
            if np.sum(grid_cell[:, conflict_index]) == 0: continue
            for t in range(1, timescale):
                X.append(grid_cell[t - 1])
                if grid_cell[t, conflict_index] > 0:
                    y.append(1)
                else:
                    y.append(0)
    X = np.array(X)
    y = np.array(y)
    print "Number of positive examples:", np.sum(y)
    print "Number of total examples:", len(y)
    return X, y

def calculate_date_diff(first_date, last_date):
    # *_date are dateutil module instances
    return (last_date.year - first_date.year)*12 + (last_date.month - first_date.month)

def calculate_grid_size(latitudes, longitudes):
    north = np.ceil(max(latitudes) * 2.0) / 2.0
    south = np.floor(min(latitudes) * 2.0) / 2.0
    east = np.ceil(max(longitudes) * 2.0) / 2.0
    west = np.floor(min(longitudes) * 2.0) / 2.0

    num_y = int((north - south) / degree_interval)
    num_x = int((east - west) / degree_interval)
    return num_x, num_y, (north, south, east, west)

def get_grid(metadata, save=False):
    latitudes = metadata['latitudes']
    longitudes = metadata['longitudes']
    deaths = metadata['deaths']
    dates = metadata['dates']

    # calculate grid size
    num_x, num_y, cardinals = calculate_grid_size(latitudes, longitudes)
    north, south, east, west = cardinals

    # calculate number of grids
    first_date, last_date = dates[0], dates[-1]
    num_grids = calculate_date_diff(first_date, last_date) + 1

    # 0 -> num_conflicts, 1 -> deaths
    grid = np.zeros((num_grids, num_x, num_y, num_features))
    for i in range(len(latitudes)):
        x = int(np.floor((longitudes[i] - west)/degree_interval))
        y = int(np.floor((latitudes[i] - south)/degree_interval))
        t = calculate_date_diff(first_date, dates[i])
        grid[t, x, y, conflict_index] += 1
        grid[t, x, y, death_index] += deaths[i]

    if save:
        np.save('../data/grid_%s' % country_name, grid)
    print "Grid shape:", grid.shape
    return grid

def get_metadata(reader):
    latitudes, longitudes, dates, deaths = [], [], [], []
    for index, row in enumerate(reader):
        if index == 0: continue
        latitudes.append(float(row[latitude_column]))
        longitudes.append(float(row[longitude_column]))
        deaths.append(int(row[death_column]))
        dates.append(parse(row[date_column], dayfirst=True))
        
    metadata = {}
    metadata['latitudes'] = latitudes
    metadata['longitudes'] = longitudes
    metadata['dates'] = dates
    metadata['deaths'] = deaths
    return metadata

# open file and make reader
f = open(csv_filename, 'rt')
reader = csv.reader(f)
# get coordinates from csv
metadata = get_metadata(reader)
# get grid from coordinates
grid = get_grid(metadata)
# obtain data samples
X, y = get_data(grid)
# gether training and test data
X_train, X_test, y_train, y_test = get_train_test(X, y)
data = (X_train, X_test, y_train, y_test)
np.save(save_name, data)
