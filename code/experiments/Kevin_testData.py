import glob
import numpy as np
import re
paths = glob.glob('data/HKUST_1F_Path2/n*_Ble.txt')
time = []
beaconNum = []
all_ap = []
for path in paths:
    with open(path, 'r') as f:
        lines = [x.strip() for x in f.readlines()]
        for line in lines[2:]:
            time.append(int(line.split()[0]))
            count = 0
            signal = re.split(r'(\ |,)',line)
            all_ap.extend(signal)

            #beaconNum.append(count)
count = 0
for ap in all_ap:
    if ap == '0C:D7:8D:7B:9F:70':
        #count += 1
        print("haha")
print(len(lines))
print(count)
print(np.mean(np.diff(np.array(time))))
print(np.std(np.diff(np.array(time))))
#print(np.mean(np.array(beaconNum)))