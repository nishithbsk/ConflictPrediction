import csv
import numpy as np
from data_preprocess import calculate_grid_size, degree_interval

csv_filename = '../data/poverty_dataset.csv'
csv_file = open(csv_filename, 'rb')
csv_reader = csv.DictReader(csv_file)

col_names = ['lat', 'lon', 
			 'households', 'urban', 'nightlights', 'consumption', 'elevation',
			 'dist_hq', 'dist_market', 'dist_road', 'dist_pop_center', 
			 'prop_metal', 'num_rooms', 'landscan_pop_est', 'asset_index']
metadata = {}
for col_name in col_names:
	metadata[col_name] = []

def value(row, col_name):
	return float(row[col_name]) if len(row[col_name]) > 0 else None

for row in csv_reader:
	if row['country'] != 'uganda': continue
	for col_name in col_names:
		metadata[col_name].append(value(row, col_name))
for key, vals in metadata.iteritems():
	avg = np.mean([x for x in vals if x != None])
	metadata[key] = [x if x != None else avg for x in vals]
latitudes, longitudes = metadata['lat'], metadata['lon']
num_x, num_y, cardinals = calculate_grid_size(metadata['lat'], metadata['lon'])
north, south, east, west = cardinals
#IMPORTANT: make sure grid boundaries are consistent with grid from data_preprocess

col_names = col_names[2:]
num_features = len(col_names)
grid = np.zeros((num_x, num_y, num_features))
counts = np.zeros((num_x, num_y))
for i in range(len(latitudes)):
    x = int(np.floor((longitudes[i] - west) / degree_interval))
    y = int(np.floor((latitudes[i] - south) / degree_interval))
    counts[x, y] += 1
    for j, feat in enumerate(col_names):
    	grid[x, y, j] += metadata[feat][i]

for x in xrange(num_x):
	for y in xrange(num_y):
		for f in xrange(num_features):
			grid[x, y, f] = grid[x, y, f] / counts[x, y]
			if counts[x, y] < 1: 
				grid[x, y, f] = 0

mask = np.sign(counts)

np.set_printoptions(precision=1, linewidth = 150, suppress=True)
print '\nCOUNTS:\n', counts
for j, feat in enumerate(col_names):
	print '\n' + feat + ':\n', grid[:, :, j]
print '\nMASK:\n', mask
# TODO(nish): grid contains the features, and mask contains the bitmask