import numpy as np
from sklearn.metrics import euclidean_distances
from base_code.analysis.overall_clusters import ChainAggregatorMethod, ChainAggregator
from base_code.analysis.metrics_base import MetricsBase, NMI, ARI
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.ticker as ticker
from base_code.algos.rap import EAP
from base_code.analysis.local_info import InfoCmaps, InfoPlotter, InfoType
#################################################

class Dataset:
    def __init__(self, **kwargs):
        self.file = kwargs['file']
        self.name = kwargs['name']

datasets = [
    Dataset(name='Aggregation', file='aggregation.txt'),
    Dataset(name='Flame', file='flame.txt'),
    Dataset(name='R15', file='R15.txt'),
    Dataset(name='Spiral', file='spiral.txt'),
    Dataset(name='Blobs', file='distant_blobs.txt'),
]

params = {
    datasets[0].name:{'preference': -18, 'q': 2.2, 'eps':-1.5},
    datasets[1].name:{'preference': -7.5, 'q':1.5, 'eps':-1},
    datasets[2].name:{'preference': -18, 'q': 0, 'eps':0},
    datasets[3].name:{'preference': -0.5, 'q': 2.8, 'eps':-1.4},
    datasets[4].name:{'preference': -20, 'q': 1.6, 'eps':-1},

}
#################################################


###############
##############
rcParams.update({'font.size': 10})
cm = plt.get_cmap('Reds')
fig, axes_all = plt.subplots(len(datasets), 4, figsize=(10, len(params)*2.5))
cols = [
    'Scatter plot\nwith local exemplars',
    'Local exemplars\nper point',
    'Inter-exemplar\nstrength',
    'Cluster\nassignments',
]
rows = [datasets[i].name for i in range(len(datasets))]
pad = 5
for ax, col in zip(axes_all[0], cols):
    # ax.set_xlabel('ss')
    ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')
for ax, row in zip(axes_all[:,0], rows):
    ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size='large', ha='center', va='center', rotation=90)

plt.tight_layout(pad=2.5, w_pad=2.5, h_pad=2)
fig.subplots_adjust(left=0.07, top=0.95)
# plt.show()
###################################################
cmaps = InfoCmaps()

for i in range(len(params)):

    current_dataset = datasets[i]
    print("############" + current_dataset.name + "############")
    data = np.loadtxt('datasets/'+current_dataset.file, dtype=float)
    X = data[:, :2]
    n = X.shape[0]
    labels_true = data[:, 2]
    cls_true = {k: np.where(labels_true == k)[0] for k in set(labels_true)}
    D = euclidean_distances(X)
    S = -D
    gnd_cls = {k: np.where(labels_true == k)[0] for k in set(labels_true)}

    ################
    row = axes_all[i]
    row[1].set_xlabel('connections')
    row[1].set_ylabel('points')
    row[2].set_xlabel('links')
    row[2].set_ylabel('pairs')

    param = params[datasets[i].name]
    print('#############################')
    print(param)
    #####################################
    #####################################
    #####################################
    #####################################
    af = EAP(**{
        'S': S,
        'preference': param['preference'],
        'q': param['q'],
    }).run()

    MetricsBase(
        gnd_cls=gnd_cls,
        cls=af.decision.clusters
    ).compute()
    print("ARI = {:.3f}, NMI = {:.3f}".format(
        ARI.compute(gnd_cls=gnd_cls,cls=af.decision.clusters),
        NMI.compute(gnd_cls=gnd_cls,cls=af.decision.clusters),
    ))

    InfoPlotter(**{
        'fig':fig,
        'ax':row[0],
        'type':InfoType.SCATTER_CENTER_WITH_CLUSTER_TYPES,
        'S':S,
        'X':X,
        'gnd_cls': gnd_cls,
        'H': af.decision.H_hat.copy()
    }).compute()

    InfoPlotter(**{
        'fig':fig,
        'ax':row[1],
        'type':InfoType.HISTOGRAM_CONNECTIONS,
        'S':S,
        'X':X,
        'gnd_cls': gnd_cls,
        'H': af.decision.H_hat.copy()
    }).compute()

    #####################################
    #####################################

    af = EAP(**{
        'S': S,
        'preference': param['preference'],
        'q': param['q'],
        'eps': param.get('eps', 0),
    }).run()

    MetricsBase(
        gnd_cls=gnd_cls,
        cls=af.decision.clusters
    ).compute()
    print("ARI = {:.3f}, NMI = {:.3f}".format(
        ARI.compute(gnd_cls=gnd_cls,cls=af.decision.clusters),
        NMI.compute(gnd_cls=gnd_cls,cls=af.decision.clusters),
    ))

    InfoPlotter(**{
        'fig':fig,
        'ax':row[2],
        'type':InfoType.HISTOGRAM_INTER_EXEMPLAR_LINKS,
        'S':S,
        'X':X,
        'gnd_cls': gnd_cls,
        'H': af.decision.H_hat.copy()
    }).compute()
    InfoPlotter(**{
        'fig':fig,
        'ax':row[3],
        'type':InfoType.SCATTER_FINAL_WITH_EXEMPLARS,
        'S':S,
        'X':X,
        'gnd_cls': gnd_cls,
        'H': af.decision.H_hat.copy()
    }).compute()


    for j in range(len(row)):
        ax = row[j]
        start, end = ax.get_xlim()
        if j==1:
            ax.xaxis.set_ticks(np.arange(0, max(end,6), max(int(end/4), 1)))
        elif j==2 and end>1:
            ax.xaxis.set_ticks(np.arange(0, end, int(end/5)))
        else:
            ax.xaxis.set_ticks(np.arange(start, end+1e-12, (end-start)/3))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
        start, end = ax.get_ylim()
        ax.yaxis.set_ticks(np.arange(start, end+1e-12, (end-start)/3))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))

plt.savefig('toy_res.eps', format='eps', dpi=300)
plt.show()
