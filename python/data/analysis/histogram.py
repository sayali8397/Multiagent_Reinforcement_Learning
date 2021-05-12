from itertools import groupby
from ast import literal_eval
from os import listdir

import sys
import matplotlib.pyplot as plt

directory_path = sys.argv[1]
history = []
for filename in listdir(directory_path):
    with open(directory_path + filename) as f:
        lines = f.readlines()
        curr_history = [tuple('f' + filename + str(x) for x in literal_eval(line.strip())) for line in lines]
        history += curr_history

point_visit_freqs = []
for k, g in groupby(sorted(history), lambda x: x):
    point_visit_freqs.append(len(list(g)))

n_bins = max(point_visit_freqs)
n, bins, patches = plt.hist(point_visit_freqs, n_bins)

xlim = n_bins - 1
while xlim > 0 and n[xlim] < 20:
    xlim -= 1

plt.xlabel('Visit count')
plt.ylabel('Number of points')
plt.xlim([1, xlim])
plt.show()