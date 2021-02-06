import numpy as np
from enum import Enum
import pickle
import io
from contextlib import redirect_stdout
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

class MetricName(Enum):
    '''
    Read the following paper to find these metrics:
    https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-10-99
    '''

    CLUSTERING_WISE_SENSITIVITY = "CLUSTERING WISE SENSITIVITY"
    CLUSTERING_WISE_PPV = "CLUSTERING WISE PPV"
    ACCURACY = "ACCURACY"
    AVERAGE_CLUSTERING_WISE_SEPARATION = "AVERAGE CLUSTERING WISE SEPARATION"
    AVERAGE_COMPLEX_WISE_SEPARATION = "AVERAGE COMPLEX WISE SEPARATION"
    CLUSTERING_SEPARATION = "CLUSTERING SEPARATION"
    ACCURACY_PLUS_SEPARATION = "CLUSTERING PLUS SEPARATION"



class MetricsBase:
    def __init__(self, gnd_cls:dict=None, cls:dict=None, verbose=True):
        self.gnd_cls = gnd_cls
        self.cls = cls
        self.verbose = verbose

        self.num_gnd_cls = len(self.gnd_cls.keys())
        self.num_cls = len(self.cls.keys())
        self.gnd_cls_keys = sorted(list(self.gnd_cls.keys()))
        self.cls_keys = sorted(list(self.cls.keys()))
        self.T = np.zeros((self.num_gnd_cls, self.num_cls))

        self.results = None # type: Result
        
    def compute(self):
        '''
        Formulas for computation are given in the paper: https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-10-99 
        '''
        ###################################
        ## Test data, taken from paper: https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-10-99
        # self.T = np.array([[
        #     7, 0, 0, 0, 0
        # ],[
        #     0, 6, 8, 0, 0
        # ],[
        #     0, 0, 0, 14, 3
        # ],[
        #     0, 0, 0, 4, 5
        # ]])
        # Ni = np.array([7, 14, 20, 8])
        # self.num_gnd_cls = 4
        # self.num_cls = 5

        #fill up T matrix by assigning T_ij => the number of common points between ith ground truth cluster and jth estimated cluster
        for i in range(self.num_gnd_cls):
            for j in range(self.num_cls):
                self.T[i, j] = len(
                    set(self.gnd_cls[self.gnd_cls_keys[i]]).intersection(
                        set(self.cls[self.cls_keys[j]])
                    )
                )
        # Definitions given in the paper
        Ni = np.array([len(self.gnd_cls[k]) for k in self.gnd_cls_keys])
        Nj = np.array([len(self.cls[k]) for k in self.cls_keys])

        T_i = np.sum(self.T, axis=1)
        T_j = np.sum(self.T, axis=0)

        ###########
        Sn_ij = self.T / Ni[:, None]
        Sn_coi = np.max(Sn_ij, axis=1)
        Sn = np.sum(Ni * Sn_coi) / np.sum(Ni)
        ##########
        # PPV_ij = self.T / T_j[None, :]
        PPV_ij = self.T / Nj
        PPV_clj = np.max(PPV_ij, axis=0)
        PPV = np.sum(T_j * PPV_clj) / np.sum(Nj)
        # PPV = np.sum(T_j * PPV_clj) / np.sum(T_j)
        ##########
        
        Acc = np.sqrt(Sn * PPV)
        ##########
        F_row_ij = self.T / T_i[:, None]
        F_col_ij = self.T / T_j[None, :]
        # assert np.array_equal(F_col_ij, PPV_ij)
        Sep = F_col_ij * F_row_ij
        Sep_coi = np.sum(Sep, axis=1)
        Sep_clj = np.sum(Sep, axis=0)
        Sep_co = np.sum(Sep_coi) / self.num_gnd_cls
        Sep_cl = np.sum(Sep_coi) / self.num_cls
        Sep = np.sqrt(Sep_co * Sep_cl)

        self.results = Result(**{
            MetricName.CLUSTERING_WISE_SENSITIVITY.name : Sn,
            MetricName.CLUSTERING_WISE_PPV.name: PPV,
            MetricName.ACCURACY.name : Acc,
            MetricName.AVERAGE_COMPLEX_WISE_SEPARATION.name : Sep_co,
            MetricName.AVERAGE_CLUSTERING_WISE_SEPARATION.name: Sep_cl,
            MetricName.CLUSTERING_SEPARATION.name : Sep,
            # 'gnd_cls' : self.gnd_cls,
            'gnd_cls' : None,
            'cls' : self.cls
        })

        if self.verbose:
            self.results.print()
        return self



class Result:
    '''
    A simple object to encapsulate necessary results data
    '''

    def __init__(self, **kwargs):
        try:
            self.CLUSTERING_WISE_SENSITIVITY = kwargs[MetricName.CLUSTERING_WISE_SENSITIVITY.name]
            self.CLUSTERING_WISE_PPV = kwargs[MetricName.CLUSTERING_WISE_PPV.name]
            self.ACCURACY = kwargs[MetricName.ACCURACY.name]
            self.AVERAGE_CLUSTERING_WISE_SEPARATION = kwargs[MetricName.AVERAGE_CLUSTERING_WISE_SEPARATION.name]
            self.AVERAGE_COMPLEX_WISE_SEPARATION = kwargs[MetricName.AVERAGE_COMPLEX_WISE_SEPARATION.name]
            self.CLUSTERING_SEPARATION = kwargs[MetricName.CLUSTERING_SEPARATION.name]
            self.ACCURACY_PLUS_SEPARATION = kwargs[MetricName.ACCURACY.name] + \
                                            kwargs[MetricName.CLUSTERING_SEPARATION.name]

            self.GND_TRUTH_CLS = kwargs['gnd_cls']
            self.COMPUTED_CLS = kwargs['cls']

        except:
            self.CLUSTERING_WISE_SENSITIVITY = None
            self.CLUSTERING_WISE_PPV = None
            self.ACCURACY = None
            self.AVERAGE_CLUSTERING_WISE_SEPARATION = None
            self.AVERAGE_COMPLEX_WISE_SEPARATION = None
            self.CLUSTERING_SEPARATION = None
            self.ACCURACY_PLUS_SEPARATION = None

            self.GND_TRUTH_CLS = None
            self.COMPUTED_CLS = None

        self.CONVERGED = None
        self.EXTRA = None
    def print(self):
        print("{:40s} = {}".format(MetricName.CLUSTERING_WISE_SENSITIVITY.value, self.CLUSTERING_WISE_SENSITIVITY))
        print("{:40s} = {}".format(MetricName.CLUSTERING_WISE_PPV.value, self.CLUSTERING_WISE_PPV))
        print("{:40s} = {}".format(MetricName.ACCURACY.value, self.ACCURACY))
        print("{:40s} = {}".format(MetricName.AVERAGE_CLUSTERING_WISE_SEPARATION.value, self.AVERAGE_CLUSTERING_WISE_SEPARATION))
        print("{:40s} = {}".format(MetricName.AVERAGE_COMPLEX_WISE_SEPARATION.value, self.AVERAGE_COMPLEX_WISE_SEPARATION))
        print("{:40s} = {}".format(MetricName.CLUSTERING_SEPARATION.value, self.CLUSTERING_SEPARATION))
        print("{:40s} = {}".format(MetricName.ACCURACY_PLUS_SEPARATION.value, self.ACCURACY_PLUS_SEPARATION))
        # return self

    def __repr__(self):
        with io.StringIO() as buf, redirect_stdout(buf):
            self.print()
            o = buf.getvalue()
        return o

    def save(self, f_name):
        with open(f_name, 'wb') as f:
            pickle.dump(self.__dict__, f)

        return self


class NMI: #normalized mutual info
    @staticmethod
    def compute(gnd_cls, cls):
        inverted_gnd_cls = dict( (v,k) for k in gnd_cls for v in gnd_cls[k]) # find cluster_index of every point for ground clusters
        inverted_cls = dict((v, k) for k in cls for v in cls[k]) #find cluster index for every point for estimated clusters
        labels_true = [inverted_gnd_cls[k] for k in sorted(inverted_gnd_cls.keys())] #list of true labels
        labels_pred = [inverted_cls[k] for k in sorted(inverted_cls.keys())] # list of estimated labels
        return normalized_mutual_info_score(labels_true, labels_pred)

class ARI: #adjusted rand index
    @staticmethod
    def compute(gnd_cls, cls):
        inverted_gnd_cls = dict( (v,k) for k in gnd_cls for v in gnd_cls[k]) # see NMI for def
        inverted_cls = dict((v, k) for k in cls for v in cls[k]) # see NMI for def
        labels_true = [inverted_gnd_cls[k] for k in sorted(inverted_gnd_cls.keys())] # see NMI for def
        labels_pred = [inverted_cls[k] for k in sorted(inverted_cls.keys())] # see NMI for def
        return adjusted_rand_score(labels_true, labels_pred)

