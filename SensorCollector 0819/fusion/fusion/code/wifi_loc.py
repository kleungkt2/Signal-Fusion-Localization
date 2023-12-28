from sites.config import *
from model.wifi import WifiNetwork, WiFiPreprocessor
from model.inout import InOutClassifier
from util.general_util import *
from util.data_util import read_wifi_fingerprint, compute_reference_features

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

site = MtrKlb()
site_name = site.name
constraints1 = site.constraints1
constraints2 = site.constraints2
map_path = site.map_path
dataset_dir = site.dataset_dir
log_dir = os.path.join(site.log_dir, 'wifi')
if not os.path.isdir(site.log_dir):
    os.mkdir(site.log_dir)
if not os.path.isdir(log_dir):
    os.mkdir(log_dir)

gridsize = site.gridsize
batch_size = 1024
n_epochs = 500
n_layers = 1
d = 32


## Generate RPs
rp_locs = generate_line_grid(constraints1, gridsize) + generate_filled_grid(constraints2, gridsize)
rp_locs = list(set(rp_locs))
im = Image.open(map_path)
draw = ImageDraw.Draw(im)
r = 2
for j in range(0, len(rp_locs)):
    draw.ellipse((rp_locs[j][0]-r, rp_locs[j][1]-r, rp_locs[j][0]+r, rp_locs[j][1]+r), fill=(255, 0, 0, 255))
im.save(os.path.join(log_dir, '%s_grid.jpg' % site_name))


# Load data, preprocess
fp_features, fp_locs, ap_list1 = read_wifi_fingerprint(glob.glob(os.path.join(dataset_dir, 'm*_WiFi.txt')))
test_features, test_locs, ap_list2 = read_wifi_fingerprint(glob.glob(os.path.join(dataset_dir, 'n*_WiFi.txt')))
print(fp_features.shape, test_features.shape)

if os.path.exists(os.path.join(log_dir, 'preprocessor')):
    with open(os.path.join(log_dir, 'preprocessor'), 'rb') as f:
        preprocessor = pickle.load(f)
else:
    preprocessor = WiFiPreprocessor()
    preprocessor.fit(fp_features, fp_locs, ap_list1, ap_list2, rp_locs)
    preprocessor.save(log_dir)

fp_features = preprocessor.preprocess(fp_features, ap_list1).astype(np.float32)
test_features = preprocessor.preprocess(test_features, ap_list2).astype(np.float32)
print(fp_features.shape, test_features.shape)

if not os.path.exists(os.path.join(log_dir, 'inout')):
    inout = InOutClassifier()
    inout.train(fp_features)
    inout.save(log_dir)


rp_features = compute_reference_features(rp_locs, fp_features, fp_locs, gridsize=gridsize).astype(np.float32)
with open(os.path.join(log_dir, 'grid.txt'), 'w') as f:
    for rp_loc, rp_feature in zip(rp_locs, rp_features):
        f.write("%d %d %s\n" % (rp_loc[0], rp_loc[1], ' '.join([str(x) for x in rp_feature])))

train_ds = tf.data.Dataset.from_tensor_slices((fp_features, fp_locs)).shuffle(10000).batch(batch_size)
test_ds = tf.data.Dataset.from_tensor_slices((test_features, test_locs)).shuffle(10000).batch(1)

#############################################################################################################
n_feature = fp_features.shape[1]
net = WifiNetwork(n_feature, rp_locs, rp_features, n_layers=n_layers, embedding_dim=d)

train_loss = tf.keras.metrics.Mean(name='train_loss')
reg_loss = tf.keras.metrics.Mean(name='regularization_loss')
loc_loss = tf.keras.metrics.Mean(name='localization_loss')
test_accuracy = tf.keras.metrics.Mean(name='test_accuracy')

if test:
    if os.path.exists(os.path.join(log_dir, 'saved_model.pb')):
        net = tf.saved_model.load(log_dir)
else:
    # Pretrain
    for i in range(n_layers):
        pretrain_step = net.pretrain_wrapper()
        for epoch in range(n_epochs):
            train_loss.reset_states()
            reg_loss.reset_states()
            for x, y in train_ds:
                t_loss, r_loss = pretrain_step(x, i)
                train_loss(t_loss)
                reg_loss(r_loss)
            print('Epoch {}, Training Loss: {}, Reg. Loss: {}'.format(epoch+1, train_loss.result(), reg_loss.result()))

    for epoch in range(n_epochs * 2):
        train_loss.reset_states()
        reg_loss.reset_states()
        loc_loss.reset_states()
        for x, y in train_ds:
            t_loss, r_loss, a_loss = net.train_step(x, y)
            train_loss(t_loss)
            reg_loss(r_loss)
            loc_loss(a_loss)
        print('Epoch {}, Training Loss: {}, Loc. Loss: {}, Reg. Loss: {}'.format(epoch + 1, train_loss.result(), loc_loss.result(), reg_loss.result()))

    signature = net.predict.get_concrete_function(x=tf.TensorSpec(shape=[None, n_feature], dtype=tf.float32))
    tf.saved_model.save(net, log_dir, signature)

rp_embeddings = net.predict(rp_features)
predictions, truths = [], []
for x, y in test_ds:
    if y[0].numpy()[1] < 9999:
        out = net.predict(x)
        sim = tf.norm(rp_embeddings - out, axis=1)
        prob1 = tf.exp(-sim ** 2 / 2)
        prob = prob1 / tf.math.reduce_sum(prob1)
        pred = tf.math.reduce_sum(prob[:, tf.newaxis] * rp_locs, axis=0)
        # pred = rp_locs[tf.argmax(prob)]
        truth = y[0].numpy()
        acc = tf.norm(truth - pred)

        test_accuracy(acc)
        predictions.append(pred)
        truths.append(truth)

print("Accuracy: {}".format(test_accuracy.result()))
im = Image.open(map_path)
draw = ImageDraw.Draw(im)
for pred, truth in zip(predictions, truths):
    draw.line((pred[0], pred[1], truth[0], truth[1]), fill=256)
# im.show()
im.save(os.path.join(log_dir, '%s_wifi.jpg' % site_name))

np.savetxt(os.path.join(log_dir, 'weight.txt'), np.ones(len(rp_locs)))
