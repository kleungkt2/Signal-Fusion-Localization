from sites.config import *
from model.mag import MagDTWMatching
from util.general_util import *
from util.data_util import read_magnetic_fingerprint, construct_mag_fp

import os
import glob
from PIL import Image, ImageDraw
import sys


site = LTA()
site_name = site.name
constraints1 = site.constraints1
constraints2 = site.constraints2
map_path = site.map_path
dataset_dir = site.dataset_dir
log_dir = os.path.join(site.log_dir, 'mag')
if not os.path.isdir(site.log_dir):
    os.mkdir(site.log_dir)
if not os.path.isdir(log_dir):
    os.mkdir(log_dir)

gridsize = site.gridsize
r = site.scale * 9


rp_locs = generate_line_grid(constraints1, gridsize) + generate_filled_grid(constraints2, gridsize)
rp_locs = list(set(rp_locs))
fp_vals, fp_locs = read_magnetic_fingerprint(glob.glob(os.path.join(dataset_dir, 'mi*_Sensors.txt')))
test_vals, test_locs = read_magnetic_fingerprint(glob.glob(os.path.join(dataset_dir, 'na*_Sensors.txt')))

mag_fp = construct_mag_fp(fp_vals, fp_locs, rp_locs, seq_length=site.scale*3, step_size=100)
with open(os.path.join(log_dir, 'fingerprint.txt'), 'w') as f:
    for k, v in mag_fp.items():
        f.write("{} {}".format(k[0], k[1]))
        print(len(v))
        for seq in v:
            f.write(" {}".format(",".join(map(str, seq))))
        f.write("\n")


w = 100
mag = MagDTWMatching(mag_fp)

predictions, truths = [], []
print(len(test_locs))
for test_val, test_loc in zip(test_vals, test_locs):
    print(len(test_loc))
    for i in range(0, len(test_loc)-w, w):
        truth = np.array(test_loc[i+w])
        prob = mag.predict(test_val[i:i+w], truth, sigma=5e2, d_range=r)
        prob = prob / np.sum(prob)
        pred = np.sum(prob[:, np.newaxis] * rp_locs, axis=0)
        acc = np.linalg.norm(pred - truth)

        print(i, acc, pred, truth)
        predictions.append(pred)
        truths.append(truth)

im = Image.open(map_path)
draw = ImageDraw.Draw(im)
for pred, truth in zip(predictions, truths):
    draw.line((pred[0], pred[1], truth[0], truth[1]), fill=256)
# im.show()
im.save(os.path.join(log_dir, '%s_mag.jpg' % site_name))

np.savetxt(os.path.join(log_dir, 'weight.txt'), np.ones(len(rp_locs)))
