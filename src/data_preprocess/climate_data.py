# Two sets of climate data at http://www.esrl.noaa.gov/psd/data/gridded/data.UDel_AirT_Precip.html
# Air temperature: 'air.mon.mean.v401.nc'
# Precipitation: 'precip.mon.total.v401.nc'
# Latitude from 89.75 deg to -89.75 deg, 0.5 deg step
# Longitude from 0.25 deg to 359.75 deg, 0.5 deg step
# Time from Jan. 1900 to Dec. 2014, 1 month step
# Air and precipitation index order: time, latitude, longitude

import numpy as np
from netCDF4 import Dataset

#nc file data starting index
year_init = 1900
lons_init = 0
lats_init = 90

#conflict grid index, obtained from conflict map information
year_start = 1997
year_end   = 2010
lats_south = 12
lats_north = 15
lons_west  = 360 - 18
lons_east  = 360 - 10

degree_interval = 0.5
num_y     = int((lats_north - lats_south)/degree_interval)
num_x     = int((lons_east  - lons_west )/degree_interval)
num_grids = (year_end - year_start + 1) * 12
num_feat  = 2

#calculating the index in the nc raw data for the desired grid
time_ind_start = (year_start - year_init     ) * 12    
time_ind_end   = (year_end   - year_init  + 1) * 12
lats_ind_south = (lats_init  - lats_south    ) * 2
lats_ind_north = (lats_init  - lats_north    ) * 2
lons_ind_west  = (lons_west  - lons_init     ) * 2
lons_ind_east  = (lons_east  - lons_init     ) * 2

#getting data from ncfile: latitude, longitude, air temperature, and precipitation
def get_metadata(nc_fid_air, nc_fid_precip):
	lats    = nc_fid_air.variables['lat'][:]
	lons    = nc_fid_air.variables['lon'][:]
	air     = nc_fid_air.variables['air'][:]
	precip  = nc_fid_precip.variables['precip'][:]
	return lats, lons, air, precip

def get_grid(lats, lons, air, precip):
	grid        = np.zeros((num_grids, num_x, num_y, num_feat))
	data_air    = air[time_ind_start: time_ind_end, lats_ind_north:lats_ind_south, lons_ind_west:lons_ind_east]
	data_precip = precip[time_ind_start: time_ind_end, lats_ind_north:lats_ind_south, lons_ind_west:lons_ind_east]
	for i in range(0, num_grids - 1):
		temp_air1   = air[time_ind_start + i, lats_ind_north:lats_ind_south, lons_ind_west:lons_ind_east]
		temp_precip1  = precip[time_ind_start + i, lats_ind_north:lats_ind_south, lons_ind_west:lons_ind_east]
		# transpose for longitude and latitude
		temp_air1    = np.transpose(temp_air1)
		temp_precip1 = np.transpose(temp_precip1)
		# the index for latitude is reversed with conflict grid convention, fixing it here
		temp_air2    = temp_air1
		temp_precip2 = temp_precip1
		for j in range(0, num_y - 1):
			temp_air1[:, j]    = temp_air2[:, num_y - j - 1]
			temp_precip1[:, j] = temp_precip2[:, num_y - j - 1]

		#giving grid the value at time i
		#feature 0: air temperature; feature 1: precipitation
		grid[i, :, :, 0] = temp_air1
		grid[i, :, :, 1] = temp_precip1
	return grid


nc_fid_air = Dataset('../data/air.mon.mean.v401.nc', 'r')
nc_fid_precip = Dataset('../data/precip.mon.total.v401.nc', 'r')

lats, lons, air, precip = get_metadata(nc_fid_air, nc_fid_precip) 

grid = get_grid(lats, lons, air, precip)

#(x, y, z) = grid.shape
#print x, y, z
#print grid[84][8][3][0]
#print grid[84][8][3][1]
