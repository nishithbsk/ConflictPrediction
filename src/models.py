import tensorflow as tf
import prettytensor as pt
import numpy as np

def import_model(_num_timesteps, _grid_size, _batch_size):
    global num_timesteps
    global grid_size
	global batch_size
    num_timesteps = _num_timesteps
    grid_size = _grid_size
	batch_size = _batch_size
    return network()

def fc_layers(input_tensor):
    return (pt.wrap(input_tensor).
            fully_connected(256, name='common_fc1').
            fully_connected(grid_size*grid_size, activation_fn=None, name='common_fc2').
            reshape([1, grid_size, grid_size, 1])).tensor

def network_conflict(input_tensor):    
    return (pt.wrap(input_tensor).
            conv2d(3, 2, stride=1).
            conv2d(5, 5, stride=1).
            flatten()
            fully_connected(128, activation_fn=None, name='conflict_fc1')).tensor
             
def network():
    conflict_grids = tf.placeholder(tf.float32, [num_timesteps, grid_size])
	gt = tf.placeholder(tf.float32, [1, grid_size])	
	mask = tf.placeholder(tf.float32, [1, grid_size])
    #poverty_grid = tf.placeholder(tf.float32, [1, grid_size])

    assert(num_timesteps > 1)
    enc_conflicts = []
    with tf.variable_scope("model") as scope:
        with pt.defaults_scope(activation_fn=tf.nn.relu,
                               batch_normalize=True,
                               learned_moments_update_rate=0.0003,
                               variance_epsilon=0.001,
                               scale_after_normalization=True):
            enc_conflicts.append(network_conflict(conflict_grids[0]))
    
    for t in range(1, num_timesteps):
        with tf.variable_scope("model", reuse=True) as scope:
            with pt.defaults_scope(activation_fn=tf.nn.relu,
                                   batch_normalize=True,
                                   learned_moments_update_rate=0.0003,
                                   variance_epsilon=0.001,
                                   scale_after_normalization=True):
                enc_conflicts.append(network_conflict(conflict_grids[t]))

    mean_conflict = tf.reduce_mean(enc_conflicts, 0)

    '''
    with tf.variable_scope("model") as scope:
        with pt.defaults_scope(activation_fn=tf.nn.relu,
                               batch_normalize=True,
                               learned_moments_update_rate=0.0003,
                               variance_epsilon=0.001,
                               scale_after_normalization=True):
            enc_poverty = network_poverty(poverty_grid)

    feats = tf.concat(0, [mean_conflict, enc_poverty])
    '''
    feats = mean_conflict
    pred = fc_layers(feats)

    return pred, conflict_grids
    
