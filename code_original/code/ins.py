from util.fusion_util import *
from sites.config import *
from model.pdr import PDR

import matplotlib.pyplot as plt


data_idx = 0

site = MtrKlb()
site_name = site.name
map_path = site.map_path
dataset_dir = site.dataset_dir

# Load data
data = read_fusion_data(os.path.join(dataset_dir))
print(len(data), len(data[data_idx]))


pdr = PDR(site.scale)
locs = []
truths = []
x, y = data[data_idx][0]['loc']

for d in data[data_idx]:
    dl = pdr.get_travel_distance(d['acc'])
    heading, _, _, _ = pdr.get_orientation(np.mean(d['mag'], axis=0), np.mean(d['gravity'], axis=0))

    x += np.cos(heading + site.map_rot) * dl
    y += np.sin(heading + site.map_rot) * dl

    locs.append((x, y))
    truths.append(d['loc'])


im = plt.imread(map_path)
plt.figure(figsize=(12, 8))
plt.imshow(im)
# plt.xlim(4000, 6000)
# plt.ylim(2200, 1500)
plt.scatter(*zip(*locs), s=1)
plt.scatter(*zip(*truths), c='r', s=1)
plt.show()
