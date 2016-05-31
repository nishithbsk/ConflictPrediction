import tensorflow as tf
import prettytensor as pt
import numpy as np
import scipy.io as io
import argparse
import models
import sys
import os
import data_loader

from progressbar import ETA, Bar, Percentage, ProgressBar
from sklearn.metrics import precision_recall_curve, average_precision_score

np.random.seed(1234)
tf.set_random_seed(0)

parser = argparse.ArgumentParser(description='Train Network for Conflict Prediction.')
parser.add_argument('-wd', '--working_directory', help='directory for storing logs')
parser.add_argument('-sf', '--save_frequency', help='Number of epochs before saving')
parser.add_argument('--model_path', help='Stored model path')
parser.add_argument('mode', choices=('train', 'eval'), help='train or eval')
args = parser.parse_args()

# Training Constants
conflict_data_file = '../../data/processed/uganda_conflicts.npy'
poverty_grid_file = '../../data/processed/uganda_poverty_grid.npy'
poverty_mask_file = '../../data/processed/uganda_poverty_mask.npy'
learning_rate = 1e-4
batch_size = 1
num_timesteps = 4
input_size = (11, 11, 2)
max_epoch = 601
dataset_size = 201
updates_per_epoch = int(np.ceil(float(dataset_size) / float(batch_size)))

if args.working_directory:
    working_directory = args.working_directory
else:
    working_directory = 'trial/'
if args.save_frequency:
    save_frequency = args.save_frequency
else:
    save_frequency = 200
if args.model_path:
    model_path = args.model_path
else:
    model_path = 'trial/checkpoints/model.ckpt-600'

def get_loss(pred, gt, conflict_mask, poverty_mask):
    loss = tf.div(tf.reduce_sum(tf.square(tf.sub(pred, gt))), 
			      tf.constant(float(batch_size)))
    loss = tf.mul(loss, conflict_mask)
    return tf.mul(loss, poverty_mask)

def train():
    data_paths = [conflict_data_file, poverty_grid_file, poverty_mask_file]
    dataset, conflict_mask, poverty_grid, poverty_mask = data_loader.read_datasets(data_paths)
    
    with tf.device('/gpu:0'): # run on specific device
        conflict_grids, pov_grid, pred, gt = models.import_model(num_timesteps, 
						                 input_size,
                                                                 poverty_grid.shape)
        loss = get_loss(pred, gt, conflict_mask, poverty_mask)
        optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1.0)
        train = optimizer.minimize(loss=loss)

    saver = tf.train.Saver()  # defaults to saving all variables

    # logging the loss function
    loss_placeholder = tf.placeholder(tf.float32)
    tf.scalar_summary('train_loss', loss_placeholder)

    merged = tf.merge_all_summaries()

    init = tf.initialize_all_variables()
    
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        writer = tf.train.SummaryWriter(os.path.join(working_directory, 'logs'),
                sess.graph_def)
        sess.run(init)

        for epoch in range(max_epoch):
            training_loss = 0.0

            widgets = ["epoch #%d|" % epoch, Percentage(), Bar(), ETA()]
            pbar = ProgressBar(updates_per_epoch, widgets=widgets)
            pbar.start()
            for i in range(updates_per_epoch):
                pbar.update(i)
                conflict_grids_batch, gt_batch = \
					dataset.next_batch(batch_size)
                _, loss_value = sess.run([train, loss], 
			                 {conflict_grids : conflict_grids_batch,
                                          pov_grid : poverty_grid,
                                          gt : gt_batch})
                training_loss += np.sum(loss_value)

            training_loss = training_loss/(updates_per_epoch * batch_size)
            print("Loss %f" % training_loss)
            
            # save model
            if epoch % save_frequency == 0:
                checkpoints_folder = os.path.join(working_directory, 'checkpoints')
                if not os.path.exists(checkpoints_folder):
                    os.makedirs(checkpoints_folder)
                saver.save(sess, os.path.join(checkpoints_folder, 'model.ckpt'),
                           global_step=epoch)

                # save summaries
                summary_str = sess.run(merged, 
                              feed_dict={conflict_grids : conflict_grids_batch,
                                         gt : gt_batch,
                                         pov_grid : poverty_grid,
                                         loss_placeholder: training_loss})
                writer.add_summary(summary_str, global_step=epoch)

        writer.close()

def evaluate(print_grid=False):
    data_paths = [conflict_data_file, poverty_grid_file, poverty_mask_file]
    dataset, conflict_mask, poverty_grid, poverty_mask = data_loader.read_datasets(data_paths, dataset_type='test')
    with tf.device('/gpu:0'): # run on specific device
        conflict_grids, pov_grid, pred, gt = models.import_model(num_timesteps, 
						                 input_size,
                                                                 poverty_grid.shape)

    saver = tf.train.Saver() 
    
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        saver.restore(sess, model_path)

        all_pred, all_gt = [], []
        for i in range(updates_per_epoch):
            conflict_grids_batch, gt_batch = \
                                    dataset.next_batch(batch_size)
            pred_value = sess.run([pred], 
                                  {conflict_grids : conflict_grids_batch,
				   pov_grid : poverty_grid,
                                   gt : gt_batch})

	    mask = conflict_mask * poverty_mask
            pred_value = pred_value * mask
            to_remove_idxs = np.where(mask.flatten() < 1)
            pred_value = np.delete(pred_value.flatten(), to_remove_idxs)
            gt_batch = np.delete(gt_batch.flatten(), to_remove_idxs)
            assert(len(pred_value) == len(gt_batch))

            for k in range(len(pred_value)):
                all_pred.append(pred_value[k])
                all_gt.append(gt_batch[k])

            if print_grid:
                np.set_printoptions(precision=1, linewidth = 150, suppress=True)
                print('-'*80)
                print(np.squeeze(pred_value)) 
                print(np.squeeze(gt_batch))
       
        assert(len(all_pred) == len(all_gt))
    
        num_align = 0
        for i in range(len(all_pred)):
            if all_gt[i] > 0:
                if all_pred[i] > 0.5: num_align += 1
            elif all_gt[i] < 1:
                if all_pred[i] <= 0.5: num_align += 1
        print "Aligned:", float(num_align)/len(all_pred)
    
        threshold = 0.5
        precision_num, precision_denom = 0.0, 0.0
        for i in range(len(all_pred)):
            if all_gt[i] == 1:
                if all_pred[i] >= threshold:
                    precision_num += 1
                    precision_denom += 1
            else:
                 if all_pred[i] >= threshold: precision_denom += 1
    
        recall_num, recall_denom = 0.0, 0.0
        for i in range(len(all_pred)):
            if all_gt[i] == 1:
                if all_pred[i] >= threshold:
                    recall_num += 1
                    recall_denom += 1
                else:
                    recall_denom += 1

        print "Precision", float(precision_num)/precision_denom
        print "Recall", float(recall_num)/recall_denom

if __name__ == "__main__":
    if args.mode == 'train':
        train()
    elif args.mode == 'eval':
        evaluate(print_grid=False)
