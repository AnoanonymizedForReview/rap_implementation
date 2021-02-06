import numpy as np
import pickle
from enum import Enum
from itertools import product
from examples.real_datasets.threads import SimScheduler
from examples.real_datasets.ap import calc_ap
from examples.real_datasets.eap import calc_sap
from examples.real_datasets.greedy import calc_greedy
from sklearn.metrics import euclidean_distances
import pathlib

#############################

class Script(Enum):
    AP = 'ap'
    SAP = 'sap'
    GREEDY = 'greedy'



class Dataset:
    DECIMALS = 3
    def __init__(self, **kwargs):

        self.SCRIPT = kwargs['script'] # type: Script
        self.GND_CLS = kwargs['gnd_cls'] # type: dict
        self.RES_DIR = kwargs.get('res_dir', '')
        self.S = kwargs['S']

        self.random_state = kwargs.get('state', np.random.RandomState(0))
        self.P_RANGE = kwargs.get('p_range', {}) # type: dict
        self.Q_RANGE = kwargs.get('q_range', {}) # type: dict
        self.THR_RANGE = kwargs.get('thr_range', {}) # type: dict
        self.NOISE = kwargs.get('noise', 0.)

        ###########################
        assert isinstance(self.SCRIPT, Script)
        self.R = self.THR_RANGE if self.SCRIPT == Script.GREEDY else self.P_RANGE
        if self.P_RANGE != {}:
            self.ps = [
                round(pref, Dataset.DECIMALS) for pref in
                np.arange(self.P_RANGE["START"], self.P_RANGE["STOP"], self.P_RANGE["STEP"])
            ]
        if self.Q_RANGE !={}:
            self.qs = [
                round(q, Dataset.DECIMALS) for q in
                np.arange(self.Q_RANGE["START"], self.Q_RANGE["STOP"], self.Q_RANGE["STEP"])
            ]
        if self.THR_RANGE !={}:
            self.thresholds = [
                round(thr, Dataset.DECIMALS) for thr in
                np.arange(self.THR_RANGE["START"], self.THR_RANGE["STOP"], self.THR_RANGE["STEP"])
            ]

        self.scheduler = SimScheduler()

    def run(self):
        if self.SCRIPT == Script.GREEDY:
            for thr in self.thresholds:
                self.scheduler.add_simulation(
                    calc_greedy, **{'thr': thr, 'S': self.S, 'gnd_cls': self.GND_CLS}
                )

        elif self.SCRIPT == Script.AP:
            for pref in self.ps:
                self.scheduler.add_simulation(calc_ap, **{
                    'pref': pref, 'S': self.S, 'gnd_cls': self.GND_CLS})

        else:
            for pref, q in product(self.ps, self.qs):
                self.scheduler.add_simulation(calc_sap, **{
                    'pref': pref, 'q': q, 'S': self.S, 'gnd_cls':self.GND_CLS})


        self.res = {'gnd_cls': self.GND_CLS, 'algo_res': self.scheduler.start_all(chunk_size=16)}

        if self.RES_DIR !='':
            pathlib.Path(self.RES_DIR).mkdir(parents=True, exist_ok=True)
            self.RES_DIR += '/'
        with open(
            self.RES_DIR + self.SCRIPT.value +'_' + str(self.R["START"]) + ' to ' + str(self.R["STOP"]) +
            '_noise_' + str(self.NOISE) + '.pickle', 'wb') as f:
            pickle.dump(self.res, f)

