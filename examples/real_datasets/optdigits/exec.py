import numpy as np
from examples.real_datasets.exec_base import Dataset, Script
from sklearn.datasets import load_digits
from sklearn.metrics import euclidean_distances

NOISE = 0.
PREF_RANGE = {
    "START": -25,
    "STOP": 0,
    "STEP": 1
}
P2_RANGE = {
    "START": 0,
    "STOP": 2.5,
    "STEP": 0.1
}
THR_RANGE = {
    "START": 0,
    "STOP": 1,
    "STEP": 0.025
}

#############################


np.random.RandomState(0)
dataset = load_digits()
X = dataset.data/16
labels_true = dataset.target
cls_true={k:np.where(labels_true==k)[0] for k in set(labels_true)}
N = X.shape[0]
S = -euclidean_distances(X)
S += np.random.randint(-5000, 5000, (N, N))/5000 * NOISE


optdigits_ds = Dataset(**{
    'script': Script.GREEDY,
    'gnd_cls': cls_true,
    'S': S,
    'res_dir': '',
    'p_range': PREF_RANGE,
    'q_range': P2_RANGE,
    'thr_range': THR_RANGE
}).run()

