import numpy as np
from scipy.spatial.distance import cdist
from collections import defaultdict
import matplotlib.pyplot as pp

fname = 'semeval2016-task3-taskB-train.tsv'
data_raw = open(fname, 'rb').readlines()
data_raw = map(lambda x: x.replace('\n', '').split('\t'), data_raw)

qorig_meta = np.asarray(map(lambda x: np.asarray(map(lambda y: float(y), x[6:16])), data_raw))
qrel_meta = np.asarray(map(lambda x: np.asarray(map(lambda y: float(y), x[16:26])), data_raw))

labels = np.asarray(map(lambda x: 0 if x[26] == 'Irrelevant' else 1, data_raw))

distances = cdist(qorig_meta, qrel_meta, 'correlation').diagonal()


val = 0. # this is the value where you want the data to appear on the y-axis.
pp.plot(labels,distances, 'x')
pp.show()

for label, dist in zip(labels, distances):
    pass
