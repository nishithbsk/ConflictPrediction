# Two sets of climate data at http://www.esrl.noaa.gov/psd/data/gridded/data.UDel_AirT_Precip.html
# Air temperature: 'air.mon.mean.v401.nc'
# Precipitation: 'precip.mon.total.v401.nc'
# Latitude from 89.75 deg to -89.75 deg, 0.5 deg step
# Longitude from 0.25 deg to 359.75 deg, 0.5 deg step
# Time from Jan. 1900 to Dec. 2014, 1 month step
# Air and precipitation index order: time, latitude, longitude

import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

#nc file data starting index
year_init = 1900
lons_init = 0
lats_init = 90

#conflict grid index, obtained from conflict map information
year_start = 1997
year_end   = 2014
lats_south = -1.5
lats_north = 4
lons_west  = 29.5
lons_east  = 35

degree_interval = 0.5
num_y     = int((lats_north - lats_south)/degree_interval)
num_x     = int((lons_east  - lons_west )/degree_interval)
num_grids = (year_end - year_start + 1) * 12
num_feat  = 2
num_timesteps = 4

#calculating the index in the nc raw data for the desired grid
time_ind_start = (year_start - year_init     ) * 12    
time_ind_end   = (year_end   - year_init  + 1) * 12 - 1
lats_ind_south = (lats_init  - lats_south    ) * 2  - 1
lats_ind_north = (lats_init  - lats_north    ) * 2
lons_ind_west  = (lons_west  - lons_init     ) * 2
lons_ind_east  = (lons_east  - lons_init     ) * 2  - 1

def get_train_test(X, y, test_percent=0.10):
    return train_test_split(X, y, test_size=test_percent, random_state=42)

def get_data(grid):
    X = []
    timescale, num_rows, num_cols, num_features = grid.shape
    for t in range(num_timesteps, timescale):
        X.append(grid[t - num_timesteps: t])
    X = np.array(X)
    return X
 
#getting data from ncfile: latitude, longitude, air temperature, and precipitation
def get_metadata(nc_fid_air, nc_fid_precip):
    lats    = nc_fid_air.variables['lat'][:]
    lons    = nc_fid_air.variables['lon'][:]
    air     = nc_fid_air.variables['air'][:]
    precip  = nc_fid_precip.variables['precip'][:]
    return lats, lons, air, precip

def reverse_col(matrix):
    row_num, col_num = matrix.shape
    matrix_temp = np.zeros((row_num, col_num))
    for i in range(col_num):
            matrix_temp[:, i] = matrix[:, col_num - 1 - i]
    return matrix_temp

def get_grid(lats, lons, air, precip):
    grid = np.zeros((num_grids, num_x, num_y, num_feat))
    for i in range(num_grids):
        temp_air     =    air[time_ind_start + i, lats_ind_north:lats_ind_south + 1, lons_ind_west:lons_ind_east + 1]
        temp_precip  = precip[time_ind_start + i, lats_ind_north:lats_ind_south + 1, lons_ind_west:lons_ind_east + 1]
        # transpose for longitude and latitude
        temp_air    = np.transpose(temp_air)
        temp_precip = np.transpose(temp_precip)
        # the index for latitude is reversed with conflict grid convention, reversing the column index
        temp_air    = reverse_col(temp_air)
        temp_precip = reverse_col(temp_precip)
        #giving grid the value at time i
        #feature 0: air temperature
        #feature 1: precipitation
        grid[i, :, :, 0] = temp_air
        grid[i, :, :, 1] = temp_precip
    return grid

nc_fid_air = Dataset('../../data/raw/air.mon.mean.v401.nc', 'r')
nc_fid_precip = Dataset('../../data/raw/precip.mon.total.v401.nc', 'r')
# getting raw data from nc file
lats, lons, air, precip = get_metadata(nc_fid_air, nc_fid_precip) 
# getting the grid
grid = get_grid(lats, lons, air, precip)
grid_early = grid[:12]
grid = np.concatenate([grid, grid_early], 0)
X = get_data(grid)
conflict_train, conflict_test, _, _ = get_train_test(X, np.ones((len(X))))
data = (conflict_train, conflict_test)
np.save('../../data/processed/uganda_climate', data)
