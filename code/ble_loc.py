from sites.config import *
from model.wifi import WifiNetwork, WiFiPreprocessor
from model.inout import InOutClassifier
from util.general_util import *
from util.data_util import read_ble_fingerprint, compute_reference_features


import tensorflow as tf
from PIL import Image, ImageDraw
import os
import sys
import glob
import numpy as np
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


test = len(sys.argv) > 1 and sys.argv[1] == '--test'

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

site = MtrCtl()
site_name = site.name
constraints1 = site.constraints1
constraints2 = site.constraints2
map_path = site.map_path
dataset_dir = site.dataset_dir
#log_dir = os.path.join(site.log_dir, 'ble')
dataset_dir = "../data/HKUST_1F_Path2/"
dataset_dir2 = "../data/HKUST_1F_Path2_2/"
log_dir = "../data/HKUST_1F_Path2/"
#print("dataset_dir = " + dataset_dir)
#print(site.log_dir)
#if not os.path.isdir(site.log_dir):
#    os.mkdir(site.log_dir)
#if not os.path.isdir(log_dir):
#    os.mkdir(log_dir)


gridsize = site.gridsize
window_size = 3000
batch_size = 1024
n_epochs = 500
n_layers = 1
d = 8

rp_locs = generate_line_grid(constraints1, gridsize) + generate_filled_grid(constraints2, gridsize)

### Load data, preprocess
fp_features, fp_locs, beacon_list1 = read_ble_fingerprint(glob.glob('data/HKUST_1F_Path2/n*_Ble.txt'), window_size=window_size)
test_features, test_locs, beacon_list2 = read_ble_fingerprint(glob.glob('data/HKUST_1F_Path2_2/n*_Ble.txt'), window_size=window_size)
print("fp_features:")
print(fp_features)

if os.path.exists(os.path.join(log_dir, 'preprocessor')):
    with open(os.path.join(log_dir, 'preprocessor'), 'rb') as f:
        preprocessor = pickle.load(f)
else:
    preprocessor = WiFiPreprocessor()
    preprocessor.fit(fp_features, fp_locs, beacon_list1, beacon_list2, rp_locs)
    preprocessor.save(log_dir)

fp_features = preprocessor.preprocess(fp_features, beacon_list1).astype(np.float32)

test_features = preprocessor.preprocess(test_features, beacon_list2).astype(np.float32)
iii = np.sum(fp_features, axis=1) != 0
fp_features = fp_features[iii]
fp_locs = np.array(fp_locs)[iii]
iii = np.sum(test_features, axis=1) != 0
test_features = test_features[iii]
test_locs = np.array(test_locs)[iii]
print(fp_features.shape, test_features.shape)

rp_features = compute_reference_features(rp_locs, fp_features, fp_locs, gridsize=gridsize, p=10).astype(np.float32)
with open(os.path.join(log_dir, 'grid.txt'), 'w') as f:
    for rp_loc, rp_feature in zip(rp_locs, rp_features):
        f.write("%d %d %s\n" % (rp_loc[0], rp_loc[1], ' '.join([str(x) for x in rp_feature])))


predictions, truths, accuracies = [], [], []
for x, y in zip(test_features, test_locs):
    if y[1] < 9999:
        iii = np.nonzero(x)[0]
        prob1 = np.exp(-np.linalg.norm(rp_features[:, iii] - x[iii], axis=1)**2 / 0.5)
        prob = prob1 / np.sum(prob1)
        pred = np.sum(prob[:, tf.newaxis] * rp_locs, axis=0)
        print(prob1)
        # pred = rp_locs[tf.argmax(prob)]
        truth = y
        acc = np.linalg.norm(truth - pred)

        predictions.append(pred)
        truths.append(truth)
        accuracies.append(acc)
        # print(acc)

print("Accuracy: {}".format(np.mean(accuracies)))
im = Image.open(map_path)
draw = ImageDraw.Draw(im)
for pred, truth in zip(predictions, truths):
    draw.line((pred[0], pred[1], truth[0], truth[1]), fill=256)
# im.show()
im.save(os.path.join(log_dir, '%s_ble.jpg' % site_name))

np.savetxt(os.path.join(log_dir, 'weight.txt'), np.ones(len(rp_locs)))
