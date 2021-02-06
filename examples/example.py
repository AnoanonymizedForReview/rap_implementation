import numpy as np
from sklearn.metrics import euclidean_distances
from base_code.algos.rap import EAP
from base_code.algos.rap2 import Container
from base_code.analysis.local_info import InfoPlotter, InfoType
import matplotlib.pyplot as plt
from base_code.analysis.d3js_export import Dumper, Position
#################################################
class Dataset:
    def __init__(self, **kwargs):
        self.file = kwargs['file']
        self.name = kwargs['name']


USE_EAP = True #Toggle between EAP and SHAPE
current_dataset = Dataset(name='Flame', file='flame.txt')
param = {'preference': -7.5, 'q':1.5, 'eps':-1, 'p2': -1}
print("############" + current_dataset.name + "############")
#################################################

data = np.loadtxt('synthetic_datasets/datasets/'+current_dataset.file, dtype=float)
X = data[:, :2]
n = X.shape[0]
labels_true = data[:, 2]

cls_true = {k: np.where(labels_true == k)[0] for k in set(labels_true)}
D = euclidean_distances(X)
S = -D

print('#############################')
print(param)
#####################################
#####################################
if USE_EAP:
    af = EAP(**{
        'S': S,
        'preference': param['preference'],
        'q': param['q'],
        'eps': param.get('eps',0),
        'N_T': 2
    }).run()
    final_H = af.decision.H_hat.copy()
else:
    c = Container(**{
        'num_layers': param.get('layers', 2),
        'S': S,
        'preferences': [param['preference'], param['p2']],
        'qs': [param['q'], -np.inf],
        'state': np.random.RandomState(0),
        'last_layer_rigid': True,
    })
    c.create_layers()
    c_centers, c_labels = c.run()
    final_H = c.H2




dumper = Dumper(**{
    'H': final_H,
    'S': S,
    'labels': ['point'+ str(j) for j in range(n)],
    'positions': [Position(x=400+30*X[j,0], y=1000-(30*X[j,1])) for j in range(n)],
    'file_name': '../visualize/graph.json',

}).dump()

for i in InfoType.__members__.values():

    f, ax = plt.subplots()
    InfoPlotter(**{
        'fig':f,
        'ax':ax,
        'type':i,
        'S':S,
        'X':X,
        'gnd_cls':{k: np.where(labels_true == k)[0] for k in set(labels_true)},
        'H': final_H
    }).compute()
    ax.set_title(i)
    plt.show()
    plt.close()
