import numpy as np


def generate_line_grid(constraints, gridsize):
    grid_list = []
    for constraint in constraints:

        x1, x2, y1, y2 = constraint
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)
        x, y = x1, y1
        sx = -1 if x1 > x2 else 1
        sy = -1 if y1 > y2 else 1
        i = gridsize

        if dx > dy:
            err = dx / 2.0
            while x != x2:
                if i == gridsize:
                    grid_list.append((x, y))
                    i = 0
                i += 1
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx

        else:
            err = dy / 2.0
            while y != y2:
                if i == gridsize:
                    grid_list.append((x, y))
                    i = 0
                i += 1
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
    return grid_list


def generate_filled_grid(constraints, gridsize):
    ## scan line algorithm
    grid_list = []
    for constraint in constraints:
        xs, ys = constraint
        n = len(xs)
        ymin, ymax = min(ys), max(ys)

        for y in range(ymin, ymax + gridsize, gridsize):
            x = []
            j = n - 1
            for i in range(n):
                if y != max(ys[i], ys[j]):
                    if ys[i] != ys[j] and (ys[i] <= y <= ys[j] or ys[j] <= y <= ys[i]):
                        intersection = int(xs[i] + (y - ys[i]) / (ys[j] - ys[i]) * (xs[j] - xs[i]))
                        if (xs[i] <= intersection <= xs[j] or xs[j] <= intersection <= xs[i]):
                            x.append(intersection)
                j = i
            x.sort()
            # print(y, x)
            for i in range(0, len(x), 2):
                for xx in range(x[i], x[i + 1] + gridsize, gridsize):
                    grid_list.append((xx, y))

    return grid_list


def distance(a,b):
    d1 = a[0]-b[0]
    d2 = a[1]-b[1]
    return np.sqrt(d1*d1 + d2*d2)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_closest_grid(loc, grid_list):
    min_idx, min_d = -1, 999999
    for grid_i, grid in enumerate(grid_list):
        dist = distance(grid, loc)
        if dist < min_d:
            min_d = dist
            min_idx = grid_i

    return min_idx


def paa(seq, n=30):
    if len(seq) % n == 0:
        subarrays = np.array_split(seq, n)
        res = np.array([item.mean() for item in subarrays]).tolist()
    else:
        value_space = np.arange(0, len(seq) * n)
        output_index = value_space // len(seq)
        input_index = value_space // n
        uniques, n_uniques = np.unique(output_index, return_counts=True)
        res = [np.array(seq)[indices].sum() / len(seq) for indices in np.split(input_index, n_uniques.cumsum())[:-1]]
    return res


def smooth(seq, interval=200):
    # seq = [(v1,ts),...]
    smoothed = []
    for i, x in enumerate(seq):
        sub_array = []
        j = i
        while j < len(seq) and  x[1] <= seq[j][1] <= x[1] + interval:
            sub_array.append(seq[j][0])
            j += 1
        smoothed.append(np.mean(sub_array, axis=0).tolist())
    return smoothed
