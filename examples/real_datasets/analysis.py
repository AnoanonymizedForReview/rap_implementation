import numpy as np
import pickle
from os import listdir
from base_code.analysis.metrics_base import MetricsBase, MetricName, Result, NMI, ARI
from examples.real_datasets.exec_base import Script

#########################################
DIR = 'optdigits'
# DIR = 'mnist'
# DIR = 'proteins'
########################################
files = [f for f in listdir(DIR)]
if DIR == '.':
    DIR = ''
else:
    DIR = DIR.strip('/')+'/'

all_files = {
    'sap': [DIR+f for f in files if f.startswith('sap') and f.endswith('.pickle')],
    'ap': [DIR+f for f in files if f.startswith('ap') and f.endswith('.pickle')],
    'greedy': [DIR+f for f in files if f.startswith('greedy') and f.endswith('.pickle')]
}


gnd_cls = {}
best_results = {}
for ftype, files in all_files.items():
    if ftype not in best_results.keys():
        best_results[ftype] = {}
    for file in files:
        with open(file, 'rb') as fh:
            res = pickle.load(fh)
            if gnd_cls == {}:
                gnd_cls = res['gnd_cls']
            res = res['algo_res'] # type: list[Result]
            # scores = [res[i].ACCURACY_PLUS_SEPARATION for i in range(len(res))]
            scores = {
            'acc' : [res[i].ACCURACY for i in range(len(res))],
            'nmi' : [NMI.compute(gnd_cls,res[i].COMPUTED_CLS) for i in range(len(res))],
            'ari' : [ARI.compute(gnd_cls,res[i].COMPUTED_CLS) for i in range(len(res))]
            }
            # scores = nmi
            # best_score = max(scores)
            # best_res = res[scores.index(best_score)]
            best_score_index = {k:np.argmax(scores[k]) for k in scores.keys()}
            # best_score = {k: scores[k][best_score_index[k]] for k in scores.keys()}
            # best_res = {k: res[best_score_index[k]] for k in scores.keys()}
            best_score ={}
            best_res = {}
            for k in best_score_index.keys():
                best_score[k] = scores[k][best_score_index[k]]
                best_res[k] = res[best_score_index[k]]
                try:
                    best_res[k].EXTRA[k] = best_score[k]
                except:
                    best_res[k].EXTRA = {}
                    best_res[k].EXTRA[k] = best_score[k]

            print({ftype: best_score})
            print("##########################")
            noise = float(file.split('noise_')[1].split('.pickle')[0])
            best_results[ftype][noise] = best_res


# print(best_results)
print('done')