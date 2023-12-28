
import json
import numpy as np
from collections import OrderedDict
from .general_util import distance, get_closest_grid, paa


def read_wifi_fingerprint(paths):
    fp_features, fp_locs, ap_list = [], [], set()

    for path in paths:
        print(path)
        with open(path, 'r') as f:
            lines = [x.strip() for x in f.readlines()]

        points = lines[1].split()
        points = [tuple(map(float, p.split(','))) for p in points]
        timestamps = list(map(int, lines[2].split()))

        idx = -1
        for line in lines[3:]:
            currT, *readings = line.split(' ')
            currT = int(currT)

            while currT >= timestamps[idx + 1]:
                idx += 1
                dx = points[idx + 1][0] - points[idx][0]
                dy = points[idx + 1][1] - points[idx][1]

            ratio = (currT - timestamps[idx]) / (timestamps[idx + 1] - timestamps[idx])
            x = points[idx][0] + ratio * dx
            y = points[idx][1] + ratio * dy

            max_r = -100
            for r in readings:
                ap, reading = r.split(',')
                ap_list.add(ap.lower())
                max_r = float(reading) if float(reading) > max_r else max_r

            if max_r > -70:
                fp_locs.append((x, y))
                fp_features.append(readings)

    ap_list = list(sorted(ap_list))

    for i, fp_feature in enumerate(fp_features):
        feature = [0 for _ in ap_list]
        for r in fp_feature:
            ap, reading = r.split(',')
            if ap in ap_list:
                feature[ap_list.index(ap.lower())] = float(reading)
        fp_features[i] = feature

    return np.array(fp_features), fp_locs, ap_list


def read_magnetic_fingerprint(paths):
    v, l = [], []
    for path in paths:
        locs, vals = [], []
        with open(path, 'r') as f:
            lines = [x.strip() for x in f.readlines()]

        points = lines[1].split()
        points = [tuple(map(float, p.split(','))) for p in points]
        timestamps = list(map(int, lines[2].split()))

        i = 0
        for line in lines[3:]:
            line = json.loads(line)
            if line['type'] == 2:
                while line['timestamp'] > timestamps[i+1]:
                    i += 1

                ratio = (line['timestamp'] - timestamps[i]) / (timestamps[i + 1] - timestamps[i])
                loc = (points[i][0] + ratio * (points[i + 1][0] - points[i][0]),
                       points[i][1] + ratio * (points[i + 1][1] - points[i][1]))
                val = np.linalg.norm(line['values'])
                locs.append(loc)
                vals.append(val)
                # vals.append((val, line['timestamp']))

        # smoothed_vals = piecewise_aggregate_approximation(vals)
        # print(len(smoothed_vals), len(vals))
        # v.append(smoothed_vals)
        v.append(vals)
        l.append(locs)

    return v, l


def construct_mag_fp(v, l, rp_locs, seq_length=40, step_size=100):
    fp = OrderedDict([(tuple(x), []) for x in rp_locs])
    for vals, locs in zip(v, l):
        for i in range(0, len(vals), step_size):
            start = locs[i]
            seq = []
            for j in range(i, len(vals)):
                val, loc = vals[j], locs[j]
                if distance(start, loc) > seq_length:
                    idx = get_closest_grid(loc, rp_locs)
                    smoothed_seq = paa(seq)
                    fp[tuple(rp_locs[idx])].append(smoothed_seq)
                    break
                seq.append(val)
    return fp


def read_lbs_data(path, ap_list=None):
    lines = open(path).readlines()

    fp_features, fp_locs, all_ap = [], [], set()
    for line in lines:
        loc, *readings = line.strip().split()
        loc = tuple(map(float, loc.split(',')))

        features = []
        for r in readings:
            mac, rssi, *_ = r.split(',')
            features.append((mac.lower(), float(rssi)))
            all_ap.add(mac.lower())

        fp_features.append(features)
        fp_locs.append(loc)

    all_ap = list(sorted(all_ap))
    if ap_list is None:
        ap_list = all_ap

    for i, fp_feature in enumerate(fp_features):
        feature = [0 for _ in ap_list]
        for r in fp_feature:
            if r[0] in ap_list:
                feature[ap_list.index(r[0])] = r[1]
        fp_features[i] = feature

    return np.array(fp_features), fp_locs, all_ap


def compute_reference_features(rp_locs, fp_features, fp_locs, gridsize=15, p=3):
    rp_features = []
    for rp in rp_locs:
        feature = np.zeros(fp_features.shape[1])
        w = 0
        for fp_feature, fp_loc in zip(fp_features, fp_locs):
            dis = max(distance(rp, fp_loc), 1)
            if dis < p * gridsize:
                feature += 1 / dis / dis * fp_feature
                w += 1 / dis / dis
        if w > 0:
            feature = feature / w

        rp_features.append(feature)

    return np.array(rp_features)


def read_ins_data(files):
    paths = []
    for file in files:
        path = []
        lines = open(file).readlines()
        for line in lines[3:]:
            line = json.loads(line)
            if line['type'] in {1, 2, 3, 4, 9}:
                path.append(line)
        paths.append(path)
    return paths


def read_ble_fingerprint(paths, beacon_list=None, window_size=1000):
    fp_features, fp_locs, all_beacon = [], [], set()
    
    for path in paths:
        with open(path, 'r') as f:
            print("in ble")
            print(path)
            lines = [x.strip() for x in f.readlines()]

        points = lines[1].split()
        print("points in read_ble")
        print(points)
        points = [tuple(map(float, p.split(','))) for p in points]
        timestamps = list(map(int, lines[2].split()))

        idx = -1
        for line in lines[3:]:
            currT, *readings = line.split(' ')
            currT = int(currT)

            while currT >= timestamps[idx + 1]:
                idx += 1
                dx = points[idx + 1][0] - points[idx][0]
                dy = points[idx + 1][1] - points[idx][1]

            ratio = (currT - timestamps[idx]) / (timestamps[idx + 1] - timestamps[idx])
            x = points[idx][0] + ratio * dx
            y = points[idx][1] + ratio * dy
            # print(x,y)

            max_r = -100
            filtered_readings = []
            for r in readings:
                uuid, major, minor, reading, last_scan_time = r.split(',')
                if currT - int(last_scan_time) < window_size:
                    all_beacon.add(uuid.lower() + ',' + major + ',' + minor)
                    max_r = float(reading) if float(reading) > max_r else max_r
                    filtered_readings.append(r)

            if max_r > -100:
                fp_locs.append((x, y))
                fp_features.append(filtered_readings)

    all_beacon = list(sorted(all_beacon))
    if beacon_list is None:
        beacon_list = all_beacon

    for i, fp_feature in enumerate(fp_features):
        feature = [0 for _ in beacon_list]
        for r in fp_feature:
            uuid, major, minor, reading, _ = r.split(',')
            key = uuid.lower() + ',' + major + ',' + minor
            if key in beacon_list:
                feature[beacon_list.index(key)] = float(reading)
        fp_features[i] = feature

    return np.array(fp_features), fp_locs, all_beacon