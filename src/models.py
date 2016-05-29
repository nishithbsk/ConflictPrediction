import tensorflow as tf
import prettytensor as pt
import numpy as np

def network():
	conflict_grids = tf.placeholder(tf.float32, [num_timesteps, grid_size])
	poverty_grid = tf.placeholder(tf.float32, [1, grid_size])

	assert(num_timesteps > 1)
	enc_conflicts = []
        with tf.variable_scope("model") as scope:
	    with pt.defaults_scope(activation_fn=tf.nn.relu,
				   batch_normalize=True,
				   learned_moments_update_rate=0.0003,
				   variance_epsilon=0.001,
				   scale_after_normalization=True):
		enc_conflicts.append(cnn_conflict(conflict_grids[0]))
	
	for t in range(1, num_timesteps):
            with tf.variable_scope("model", reuse=True) as scope:
                with pt.defaults_scope(activation_fn=tf.nn.relu,
				       batch_normalize=True,
				       learned_moments_update_rate=0.0003,
				       variance_epsilon=0.001,
				       scale_after_normalization=True):
		    enc_conflicts.append(cnn_conflict(conflict_grids[t]))

	mean_conflict = tf.reduce_mean(enc_conflicts, 0)

        with tf.variable_scope("model") as scope:
	    with pt.defaults_scope(activation_fn=tf.nn.relu,
				   batch_normalize=True,
				   learned_moments_update_rate=0.0003,
				   variance_epsilon=0.001,
				   scale_after_normalization=True):
                enc_poverty = cnn_poverty(poverty_grid)

        feats = tf.concat(0, [mean_conflict, enc_poverty])
        pred = fc_layers(feats)

        return pred, conflict_grids, poverty_grid 
	
