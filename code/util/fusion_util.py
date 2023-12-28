import os
import glob
import pickle
import json
import tensorflow as tf
import numpy as np
from collections import OrderedDict

from model.mag import MagDTWMatching
from util.general_util import smooth

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def location_sim(signal, site, rp_locs):
    g = lambda x, loc, accuracy: np.exp(-((x[0]-loc[0])**2 + (x[1]-loc[1])**2) / (2*(accuracy * accuracy / 2.27)))
    coord = [signal[0], signal[1], 1]
    loc = np.matmul(coord, site.transformation)
    return [g(x, loc, signal[2]*site.scale) for x in rp_locs], loc


def read_reference_points(grid_path):
    rp_locs, rp_features = [], []
    with open(grid_path) as f:
        lines = f.readlines()
    for line in lines:
        x, y, *features = line.split(' ')
        features = list(map(float, features))
        rp_locs.append((int(x),int(y)))
        rp_features.append(features)

    return rp_locs, rp_features


def read_magnetic_fingerprint(path):
    with open(path) as f:
        lines = [x.strip() for x in f.readlines()]

    fp = OrderedDict()
    for line in lines:
        x, y, *seqs = line.split()
        s = []
        for seq in seqs:
            s.append(list(map(float, seq.split(','))))
        fp[(float(x), float(y))] = s
    return fp


def load_mag_model(path):
    mag_fp = read_magnetic_fingerprint(os.path.join(path, 'fingerprint.txt'))
    mag = MagDTWMatching(mag_fp)
    mag_w = np.loadtxt(os.path.join(path, 'weight.txt'))
    return mag, mag_w


def load_tf_model(path):
    weight = np.loadtxt(os.path.join(path, 'weight.txt'))
    model = tf.saved_model.load(path)
    return model, weight


def load_pickle_model(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model


def read_fusion_data(directory, name='na*', window_size=1000):
    sensor_lookup = {1: 0, 2: 1, 3: 2, 4: 3, 9: 4}
    wifi_files = sorted(glob.glob(os.path.join(directory, '%s_WiFi.txt' % name)))
    sensor_files = sorted(glob.glob(os.path.join(directory, '%s_Sensors.txt' % name)))
    google_files = sorted(glob.glob(os.path.join(directory, '%s_Google.txt' % name)))
    gps_files = sorted(glob.glob(os.path.join(directory, '%s_GPS.txt' % name)))

    paths = []
    for w, s, g, gg in zip(wifi_files, sensor_files, google_files, gps_files):
        print(w)
        path = []
        lines = open(w).readlines()
        for i in range(3, len(lines)):
            ts, readings = lines[i].split(maxsplit=1)
            lines[i] = "%s wifi %s" % (ts, readings)

        lines2 = []
        for line in open(s).readlines()[3:]:
            line = json.loads(line)
            if line["type"] in {1, 2, 3, 4, 9}:
                lines2.append("%d sensor %d %s" % (line["timestamp"], line["type"], ','.join(map(str, line["values"]))))

        lines3 = []
        for line in open(g).readlines()[1:]:
            line = json.loads(line)
            lines3.append(
                "%d google %f,%f,%f" % (line["timestamp"], line["latitude"], line["longitude"], line["accuracy"]))

        lines4 = []
        for line in open(gg).readlines()[1:]:
            line = json.loads(line)
            lines4.append(
                "%d gps %f,%f,%f" % (line["timestamp"], line["latitude"], line["longitude"], line["accuracy"]))

        points = lines[1].strip().split()
        points = [tuple(map(float, p.split(','))) for p in points]
        timestamps = list(map(int, lines[2].strip().split()))
        lines = list(sorted(lines[3:] + lines2 + lines3 + lines4))
        del lines2, lines3, lines4

        i = -1
        last_ts = timestamps[0]
        sensor_tmp = [[], [], [], [], []]
        google_tmp = None
        gps_tmp = None
        wifi_tmp = None
        for idx in range(len(lines)):
            ts, stype, *tmp = lines[idx].split()
            ts = int(ts)

            if ts > timestamps[i + 1]:
                i += 1

            if stype == 'sensor':
                if tmp[0] in {'1', '2', '3', '4', '9'}:
                    sensor_tmp[sensor_lookup[int(tmp[0])]].append((list(map(float, tmp[1].split(','))), ts))
            elif stype == 'google':
                google_tmp = list(map(float, tmp[0].split(',')))
            elif stype == 'wifi':
                if wifi_tmp is None or len(tmp) > len(wifi_tmp):
                    wifi_tmp = tmp
            elif stype == 'gps':
                gps_tmp = list(map(float, tmp[0].split(',')))

            if ts - last_ts >= window_size:
                ## pack all data
                ratio = (ts - timestamps[i]) / (timestamps[i + 1] - timestamps[i])
                loc = (points[i][0] + ratio * (points[i + 1][0] - points[i][0]),
                       points[i][1] + ratio * (points[i + 1][1] - points[i][1]))

                d = {'timestamp': ts, 'loc': loc}
                if gps_tmp is not None:
                    d['gps'] = gps_tmp

                if google_tmp is not None:
                    d['google'] = google_tmp

                if wifi_tmp is not None:
                    ap_list, rssi_vec = [], []
                    for t in wifi_tmp:
                        mac, rssi = t.split(',')
                        ap_list.append(mac.lower())
                        rssi_vec.append(float(rssi))
                    if max(rssi_vec) > -70:
                        d['wifi'] = (np.array([rssi_vec]), ap_list)

                for j, k in enumerate(['acc', 'mag', 'heading', 'gyro', 'gravity']):
                    if len(sensor_tmp[j]) > 0:
                        d[k] = smooth(sensor_tmp[j])

                path.append(d)
                sensor_tmp = [[], [], [], [], []]
                google_tmp = None
                gps_tmp = None
                wifi_tmp = None
                last_ts = ts

        paths.append(path)

    return paths
