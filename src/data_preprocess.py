import os
import sys
import csv
import dateutil
import numpy as np
import matplotlib.pyplot as plt

from dateutil.parser import parse
from dateutil.relativedelta import relativedelta

csv_filename = '../data/liberia.csv'
latitude_column = 19
longitude_column = 20
date_column = 3
degree_interval = 0.5

def calculate_date_diff(first_date, last_date):
    # *_date are dateutil module instances
    return (last_date.year - first_date.year)*12 + (last_date.month - first_date.month)

def calculate_grid_size(latitudes, longitudes):
    # args are lists of floats
    north = np.ceil(max(latitudes))
    south = np.floor(min(latitudes))
    east = np.ceil(max(longitudes))
    west = np.floor(min(longitudes))

    num_y = int((north - south)/degree_interval)
    num_x = int((east - west)/degree_interval)
    return num_x, num_y, (north, south, east, west)

def get_grid(metadata):
    latitudes = metadata['latitudes']
    longitudes = metadata['longitudes']
    dates = metadata['dates']

    # calculate grid size
    num_x, num_y, cardinals = calculate_grid_size(latitudes, longitudes)
    north, south, east, west = cardinals

    # calculate number of grids
    first_date, last_date = dates[0], dates[-1]
    num_grids = calculate_date_diff(first_date, last_date) + 1

    grid = np.zeros((num_grids, num_x, num_y))
    for latitude, longitude, date in zip(latitudes, longitudes, dates):
        x = int(np.ceil((longitude - west)/degree_interval))
        y = int(np.ceil((latitude - south)/degree_interval))
        t = calculate_date_diff(first_date, date)
        grid[t, x, y] += 1
    return grid

def get_metadata(reader):
    latitudes, longitudes, dates = [], [], []
    for index, row in enumerate(reader):
        if index == 0: continue
        latitudes.append(float(row[latitude_column]))
        longitudes.append(float(row[longitude_column]))
        dates.append(parse(row[date_column], dayfirst=True))

    metadata = {}
    metadata['latitudes'] = latitudes
    metadata['longitudes'] = longitudes
    metadata['dates'] = dates
    return metadata

# open file and make reader
f = open(csv_filename, 'rt')
reader = csv.reader(f)
# get coordinates from csv
metadata = get_metadata(reader)
# get grid from coordinates
grid = get_grid(metadata)
print grid.shape
