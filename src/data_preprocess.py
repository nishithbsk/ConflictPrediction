import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

csv_filename = '../data/liberia.csv'
latitude_column = 19
longitude_column = 20

def get_grid(latitudes, longitudes):
    north = np.ceil(max(latitudes))
    south = np.floor(min(latitudes))
    east = np.ceil(max(longitudes))
    west = np.floor(min(longitudes))

    num_y = (north - south)/0.5
    num_x = (east - west)/0.5
    print "num x:", num_x
    print "num y:", num_y
    grid = np.zeros((num_x, num_y))
    
    for latitude, longitude in zip(latitudes, longitudes):
        x = np.ceil((longitude - west)/0.5)
        y = np.ceil((latitude - south)/0.5)
        print x, y
        grid[x, y] += 1
    print grid

def get_latitudes_longitudes(reader):
    latitudes, longitudes = [], []
    for index, row in enumerate(reader):
        if index == 0: continue
        latitudes.append(float(row[latitude_column]))
        longitudes.append(float(row[longitude_column]))
    return latitudes, longitudes

# open file and make reader
f = open(csv_filename, 'rt')
reader = csv.reader(f)
# get coordinates from csv
latitudes, longitudes = get_latitudes_longitudes(reader)
# get grid from coordinates
get_grid(latitudes, longitudes)

