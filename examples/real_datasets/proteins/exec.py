import numpy as np
from examples.real_datasets.exec_base import Dataset,Script
import pickle


NOISE = 0.7
PREF_RANGE = {
    "START": -0.5,
    "STOP": 1,
    "STEP": 0.1
}
P2_RANGE = {
    "START": -2,
    "STOP": 0.5,
    "STEP": 0.05
}
THR_RANGE = {
    "START": 0,
    "STOP": 1,
    "STEP": 0.025
}

#############################
class WeightedSimBuilder:
    pass

with open('sim_builder.pickle', 'rb') as f:
    sim_builder = pickle.load(f)

S = sim_builder.S
N = S.shape[0]
np.random.RandomState(0)
S += np.random.randint(-5000, 5000, (N, N))/5000 * NOISE
cls_true = {
    sim_builder.complexes_set.index(k):[
        sim_builder.proteins_set.index(v) for v in ps
    ]
    for k,ps in sim_builder.c_p.items()
}


proteins_ds = Dataset(**{
    'script': Script.GREEDY,
    'gnd_cls': cls_true,
    'S': S,
    'res_dir': '',
    'p_range': PREF_RANGE,
    'q_range': P2_RANGE,
    'thr_range': THR_RANGE,
    'noise' : NOISE
}).run()

