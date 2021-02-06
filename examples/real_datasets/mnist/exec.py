import numpy as np
from examples.real_datasets.exec_base import Dataset,Script
from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle
from sklearn.metrics import euclidean_distances
#####################################
NOISE = 0.
PREF_RANGE = {
    "START": -25,
    "STOP": -5,
    "STEP": 0.5
}
P2_RANGE = {
    "START": 6,
    "STOP": 8,
    "STEP": 0.1
}
THR_RANGE = {
    "START": 0,
    "STOP": 1,
    "STEP": 0.025
}

#############################

state = np.random.RandomState(0)
mnist = fetch_mldata('MNIST original')
N=200
# idx = range(60000,60000+N)
mnist_sample_data, labels_true = shuffle(mnist.data, mnist.target, random_state=state, n_samples=N)
mnist_sample_data = mnist_sample_data/255

cls_true={k:np.where(labels_true==k)[0] for k in set(labels_true)}
S = -euclidean_distances(mnist_sample_data)

S += np.random.randint(-5000, 5000, (N, N))/5000 * NOISE


mnist_ds = Dataset(**{
    'script': Script.SAP,
    'gnd_cls': cls_true,
    'S': S,
    'res_dir': '',
    'p_range': PREF_RANGE,
    'q_range': P2_RANGE,
    'thr_range': THR_RANGE
}).run()
