from util.fusion_util import *
from util.general_util import distance, paa
from sites.config import *
from model.bayes_filter import ParticleFilter
from model.pdr import PDR


import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

site = LTA()
site_name = site.name
map_constraints = site.in_constraints, site.out_constraints
map_path = site.map_path
dataset_dir = site.dataset_dir
log_dir = site.log_dir

data_idx = 0
N = 1000
gridsize = site.gridsize
sigma_w = 5
sigma_m = 1e4
r = site.scale * 9



# Load model, rp, fingerprint
mag, mag_w = load_mag_model(os.path.join(log_dir, 'mag'))
wifi, wifi_w = load_tf_model(os.path.join(log_dir, 'wifi'))
preprocessor = load_pickle_model(os.path.join(log_dir, 'wifi', 'preprocessor'))
inout = load_pickle_model(os.path.join(log_dir, 'wifi', 'inout'))

rp_locs, wifi_rpf = read_reference_points(os.path.join(log_dir, 'wifi', 'grid.txt'))
out_locs = []  # generate_filled_grid([([2070, 2400, 2400, 2070], [1900, 1900, 2300, 2300])], gridsize)
area_locs = rp_locs + out_locs

wifi_w = np.hstack((wifi_w, np.ones(len(out_locs))))
mag_w = np.hstack((mag_w, np.ones(len(out_locs))))
wifi_rpf = wifi.predict(wifi_rpf)


# Load data
data = read_fusion_data(dataset_dir)
print(len(data), len(data[data_idx]))


# Particle Filter
pf = ParticleFilter(N, area_locs, map_constraints)
pdr = PDR(site.scale)
fig = plt.figure(figsize=(12, 8))
im = plt.imread(map_path)
plt.imshow(im)
# plt.xlim(0, 500)
# plt.ylim(800, 500)
scat = plt.scatter(pf.particles[:, 0], pf.particles[:, 1], s=3)
curr = plt.scatter(0, 0, s=20, c='r')
truth = plt.scatter(0, 0, s=20, c='lightgreen')

mag_seq = []
isin = 0
last_loc = None


def animate(i):
    global isin, mag_seq, last_loc
    d = data[data_idx][i]
    loc = d['loc']
    start_time = time.time()

    dl = pdr.get_travel_distance(d['acc'])
    heading, _, _, _ = pdr.get_orientation(np.mean(d['mag'], axis=0), np.mean(d['gravity'], axis=0))
    pf.predict(heading + site.map_rot, dl)
    scat.set_offsets(pf.particles[:, :2])

    sim = np.ones(len(area_locs))
    if 'wifi' in d:
        wifi_feature = preprocessor.preprocess(*d['wifi']).astype(np.float32)
        isin = inout.predict(wifi_feature.ravel())
        if isin > 0:
            wifi_sim = np.exp(-tf.norm(wifi_rpf - wifi.predict(wifi_feature), axis=1) ** 2 / (2 * sigma_w * sigma_w))
            wifi_sim = np.hstack((wifi_sim, np.zeros(len(out_locs))))
            sim *= wifi_sim

    if 'mag' in d and isin > 0:
        mag_seq = (mag_seq + np.linalg.norm(d['mag'], axis=1).tolist())[-100:]
        mag_sim = mag.predict(paa(mag_seq), last_loc, sigma=sigma_m, d_range=r)
        sim *= mag_sim

    if 'google' in d:
        google_sim, g_loc = location_sim(d['google'], site, area_locs)
        sim *= google_sim

    if 'gps' in d:
        gps_sim, _ = location_sim(d['gps'], site, area_locs)
        sim *= gps_sim

    pf.update(sim)
    last_loc = pf.estimate()
    curr.set_offsets(last_loc)
    truth.set_offsets(loc)
    print(last_loc, loc, distance(last_loc, loc) / site.scale, time.time()-start_time)

    if pf.neff() < N / 2:
        pf.resample()

    return scat, curr, truth


anim = animation.FuncAnimation(fig, animate, frames=len(data[data_idx]), interval=400, blit=True, repeat=False)
plt.show()
# anim.save('123.gif', writer='pillow')
