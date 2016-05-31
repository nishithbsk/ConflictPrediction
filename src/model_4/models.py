import tensorflow as tf
import prettytensor as pt
import numpy as np

def import_model(_num_timesteps, _conflict_grid_size, _poverty_grid_size):
    global num_timesteps
    global conflict_grid_size
    global poverty_grid_size
    num_timesteps = _num_timesteps
    conflict_grid_size = _conflict_grid_size
    poverty_grid_size = _poverty_grid_size
    return network()

def fc_layers(input_tensor, size):
    return (pt.wrap(input_tensor).
            fully_connected(256, name='common_fc1').
            fully_connected(size*size, activation_fn=tf.sigmoid, name='common_fc2').
            reshape([size, size])).tensor

def network_poverty(input_tensor):
    return (pt.wrap(input_tensor).
            conv2d(3, 2, stride=1, name='poverty_conv1').
            conv2d(5, 5, stride=1, name='poverty_conv2').
            flatten().
            fully_connected(64, activation_fn=None, name='poverty_fc1')).tensor

def network_conflict(input_tensor):    
    return (pt.wrap(input_tensor).
            conv2d(3, 2, stride=1, name='conflict_conv1').
            conv2d(5, 5, stride=1, name='conflict_conv2').
            flatten().
            fully_connected(128, activation_fn=None, name='conflict_fc1')).tensor

def network_rnn(rnn_inputs):
    return (pt.wrap(rnn_inputs).
            cleave_sequence(num_timesteps).
            sequence_lstm(64).
            squash_sequence())[num_timesteps - 1,:]
             
def network(): 
    gt = tf.placeholder(tf.float32, [conflict_grid_size[0], conflict_grid_size[1]])	
    conflict_grids = tf.placeholder(tf.float32, [num_timesteps,
                                                 conflict_grid_size[0],
                                                 conflict_grid_size[1],
                                                 conflict_grid_size[2]])
    poverty_grid = tf.placeholder(tf.float32, [poverty_grid_size[0],
                                               poverty_grid_size[1],
                                               poverty_grid_size[2],
                                               poverty_grid_size[3]])
    
    assert(num_timesteps > 1)
    with tf.variable_scope("model") as scope:
        with pt.defaults_scope(activation_fn=tf.nn.relu,
                               batch_normalize=True,
                               learned_moments_update_rate=0.0003,
                               variance_epsilon=0.001,
                               scale_after_normalization=True):
            enc_conflicts = network_conflict(conflict_grids)
            rnn_output = network_rnn(enc_conflicts)
    
    rnn_output = tf.reshape(rnn_output, [1, 64])

    with tf.variable_scope("model") as scope:
        with pt.defaults_scope(activation_fn=tf.nn.relu,
                               batch_normalize=True,
                               learned_moments_update_rate=0.0003,
                               variance_epsilon=0.001,
                               scale_after_normalization=True):
            enc_poverty = network_poverty(poverty_grid)
    
    feats = tf.concat(1, [rnn_output, enc_poverty])

    pred = fc_layers(feats, conflict_grid_size[0])

    return conflict_grids, poverty_grid, pred, gt
