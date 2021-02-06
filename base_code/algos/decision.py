import numpy as np
from base_code.analysis.overall_clusters import ChainAggregatorMethod, ChainAggregator



class Decision:
    '''
    Related to decision phase of AP, EAP and SHAPE 
    '''
    def __init__(self, **kwargs):
        self.H = kwargs['H'].copy() # type: np.ndarray # Thresholded binary variables matrix
        self.S = kwargs['S']  # type: np.ndarray #Similarity martix
        self.method = kwargs.get('method', ChainAggregatorMethod.SYMMETRIC_H) # which method to use. see ChainAggregatorMethod class for possibilities
        self.N_T = kwargs.get('N_T', 0) # pruning threshold
        self.rigid = kwargs.get('rigid', False) # if rigid is true, we fall back to the decision process of standard AP

        self.H_hat = self.H.copy()
        self.K = np.where(np.diagonal(self.H) == 1)[0] # exemplars
        self.left_outs = np.where(np.sum(self.H, axis=1)==0)[0] # points that have failed to select any exemplar
        self.closests_exemplars = self.K[np.argmax(self.S[:, self.K], axis=1)] # list of most similar exemplars for every point
        self.aggregator = None # type: ChainAggregator # instance of ChainAggregator class for finding connected components
        self.clusters = {} # final clusters will be stored in this dictionary
        self.N = self.H.shape[0] # number of points.
        self.ind = np.arange(self.N, dtype=int) # an array [0, N)


    def compute(self):
        if self.rigid: # if parameter "rigid" is true, fall back to decision phase of standard AP
            self.H_hat = np.zeros(self.H.shape, dtype=bool) # declare a fresh NxN matrix of zeros
            self.H_hat[self.K, self.K]= 1 # make diagonal equal to 1 at indices of exemplars
            self.H_hat[self.ind, self.closests_exemplars] = 1 # assign every point to its closest exemplar. So, exemplars will choose themselves as we are giving S matrix
        else: # we need decision phase of EAP
            self.ensure_consistency() # to ensure E(.) constraint
            self.include_left_outs() # assign self.left_outs to their closest exemplars
            if self.N_T > 1: # if N_T<=1, there is no sense of pruning as an exemplar pair must have at least 1 common point for pruning to be applied
                self.prune()
        #the above functions will change the H_hat matrix. It is now fed to aggregator object that finds connected components as clusters
        self.aggregator = ChainAggregator(H=self.H_hat, method=self.method)
        self.clusters = self.aggregator.get_clusters()
        return self

    def ensure_consistency(self):
        for i in list(set(self.ind) - set(self.K)): # loop through all non-exemplars
            self.H[:,i] = 0 # ensure that no point selects any non-exemplar as its representative
        return self

    def include_left_outs(self):
        for i in self.left_outs: # loop through self.left_outs
            self.H[i,self.closests_exemplars[i]] = 1 # assign each left_out point to its closest exemplar
        return self

    def prune(self):
        for i in range(len(self.K)): # loop through local exemplars
            e_i = self.K[i]   # save instance of exemplar for notational ease
            for j in range(i+1, len(self.K)): # 2nd loop through local exemplars
                e_j = self.K[j] # save instance of exemplar for notational ease
                commons = np.where(np.bitwise_and(self.H[:, e_i], self.H[:, e_j]) == 1)[0] # find common points between e_i and e_j
                if  0 < len(commons) < self.N_T: # if number of common points is less than threshold N_T
                    self.H_hat[commons[self.S[commons, e_i] > self.S[commons, e_j]], e_j] = 0 # For the points closer to e_i compared to e_j, remove their assignment from e_j
                    self.H_hat[commons[self.S[commons, e_i] < self.S[commons, e_j]], e_i] = 0 # For the points closer to e_j compared to e_i, remove their assignment from e_i
        return self

