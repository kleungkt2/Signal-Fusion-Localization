from util.fusion_util import *
from util.general_util import distance, paa
from sites.config import *
from model.bayes_filter import ParticleFilter, KalmanFilter
from model.pdr import PDR


import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import numpy as np

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

site = MtrKlb()
site_name = site.name
map_constraints = site.in_constraints, site.out_constraints
map_path = site.map_path
dataset_dir = site.dataset_dir
log_dir = site.log_dir

data_idx = 2
N = 1000
gridsize = site.gridsize
sigma_w = 5
sigma_m = 3e2
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
kf = KalmanFilter(x=[data[data_idx][0]['loc'][0], data[data_idx][0]['loc'][1],0,0], P=np.zeros((4,4)), H=np.diag([1,1,0,0]))
pdr = PDR(site.scale, site.map_rot)
fig = plt.figure(figsize=(12, 8))
im = plt.imread(map_path)
plt.imshow(im)
# plt.xlim(0, 500)
# plt.ylim(800, 500)
# plt.xlim(4000, 6000)
# plt.ylim(2200, 1500)
scat = plt.scatter(pf.particles[:, 0], pf.particles[:, 1], s=3)
curr = plt.scatter(0, 0, s=20, c='r')
truth = plt.scatter(0, 0, s=20, c='lightgreen')
speed_text = plt.text(700, 175, '', fontsize=12)

mag_seq = []
isin = 0
last_loc = data[data_idx][0]['loc']
last_ts = data[data_idx][0]['timestamp']-1000
alpha = 0.8
speed = np.zeros(2)
accuracy = []


def init_func():
    return scat, curr, truth, speed_text


def animate(i):
    global isin, mag_seq, last_loc, last_ts, speed, accuracy

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
        mag_seq = (mag_seq + np.linalg.norm(d['mag'], axis=1).tolist())[-300:]
        mag_sim = mag.predict(paa(mag_seq), last_loc, sigma=sigma_m, d_range=r)
        sim *= mag_sim

    # if 'google' in d:
    #     google_sim, g_loc = location_sim(d['google'], site, area_locs)
    #     sim *= google_sim
    #
    # if 'gps' in d:
    #     gps_sim, _ = location_sim(d['gps'], site, area_locs)
    #     sim *= gps_sim

    pf.update(sim)
    curr_loc = pf.estimate()
    curr.set_offsets(curr_loc)
    truth.set_offsets(loc)
    acc = distance(curr_loc, loc) / site.scale
    accuracy.append(acc)
    # print(last_loc, loc, distance(last_loc, loc)*3/40, time.time()-start_time)

    speed_loc = (curr_loc - last_loc) / (d['timestamp'] - last_ts) * 1e3 / site.scale
    speed_acc, delta_v = pdr.get_speed(d['acc'], d['mag'], d['gravity'], d['timestamp'], last_v=speed)
    speed = alpha * speed_acc + (1-alpha) * speed_loc
    last_loc = curr_loc
    last_ts = d['timestamp']

    # t = (d['timestamp'] - last_ts) / 1000
    # F = np.eye(4)
    # F[0][2] = F[1][3] = t
    # B = np.array([[0.5 * t * t, 0], [0, 0.5 * t * t], [t, 0], [0, t]])
    # u = pdr.get_acc(d['acc'], d['mag'], d['gravity'])
    # Q = (B @ B.T) * 2
    # R = np.diag([10,10,1,1])
    # x = kf.step(F=F, B=B, u=u, Q=Q, R=R, z=[curr_loc[0], curr_loc[1], 0, 0])
    # acc2 = distance(x[:2], loc) / site.scale
    # speed = np.linalg.norm(x[2:])
    # print(curr_loc, x, acc, acc2, speed)
    # last_loc = curr_loc
    # last_ts = d['timestamp']
    
    speed_text.set_text('Speed: {:.3f} m/s'.format(np.linalg.norm(speed)))
    # print('{}: {}, {} {:.2f}, {} {:.2f}, {} {:.2f}'.format(
    #     loc, acc, speed_loc, np.linalg.norm(speed_loc), speed_acc, np.linalg.norm(speed_acc),
    #     speed, np.linalg.norm(speed)))


    if pf.neff() < N / 2:
        pf.resample()

    return scat, curr, truth, speed_text


anim = animation.FuncAnimation(fig, animate, frames=len(data[data_idx]), interval=300, blit=True, repeat=False, init_func=init_func)
# anim.save(os.path.join(log_dir, 'result.gif'), writer='pillow')
# plt.show()

print('Localization accuracy: ', np.mean(accuracy))

for i in range(len(data[data_idx])):
    animate(i)



