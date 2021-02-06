import numpy as np
from copy import deepcopy
from base_code.algos.decision import Decision

class Container:
    """
    Manager for layers of HAP with modified I(.) constraint of EAP as well as modified constraint
    of SCAP. Principally this code can run multiple HAP layers. But we will
    stick to SHAPE with 2 layers. It initializes the layers, runs every iteration and prepares
    results in the end.
    """

    def __init__(self, **kwargs):
        """
        :param num_layers: Number of Layers
        :param S: NxN similarity matrix
        :param max_itr: Maximum allowed iterations before results generation. Default=2000
        :param conv_itr: Iterations required for convergence. Default=200
        :param damping: Damping factor for messages. Default=0.8
        :param preferences: List of preferences with size equal to num_layers.
        :param p1s: SCAP penalty. Either ignore this argument or give a list with size = num_layers
        :param qs: EAP penalty. Either ignore this argument or give a list with size = num_layers
        :param thrs: Either ignore this argument or give a list with size = num_layers. Default= [0, 0] These are the message thresholds for selecting h_{ij} = 0 OR 1
        :param state: np.random.RandomState(.) object to keep results consistent 
        :param verbose: Set to False if you don't want to see iterations in progress
        :param last_layer_rigid: Set to true if you want the last layer to use AP decision, as in case of SHAPE. It is False by default.
        :param pruning_thr: Either ignore this argument or give a list with size = num_layers. Default= [N_T=0, N_T=0]
        :param num_layers: Number of Layers
        
        Examples: 
            A sample SHAPE case
                >>> c = Container(**{
                    'num_layers': 2,
                    'S': S_MATRIX,
                    'preferences': [pref_1, pref_2],
                    'qs': [q, -np.inf],
                    'state': np.random.RandomState(0),
                    'last_layer_rigid': True,})


            A sample HAP case with 2 layers
                >>> c = Container(**{
                    'num_layers': 2,
                    'S': S_MATRIX,
                    'preferences': [pref_1, pref_2],
                    'state': np.random.RandomState(0),
                    'last_layer_rigid': True,})


            An AP case
                >>> c = Container(**{
                    'num_layers': 1,
                    'S': S_MATRIX,
                    'preferences': [pref_1],
                    'state': np.random.RandomState(0),
                    'last_layer_rigid': True,})
        
        """

        self.L = kwargs['num_layers']
        self.S = np.array(kwargs['S'])
        self.max_itr = kwargs.get('max_itr', 2000)
        self.conv_itr = kwargs.get('conv_itr', 200)
        self.damping = kwargs.get('damping', 0.8)
        self.preferences = list(kwargs['preferences'])
        self.p1s = kwargs.get('p1s', [np.inf]*self.L)
        self.qs = kwargs.get('qs', [-np.inf]*self.L)
        self.thrs = kwargs.get('thrs', [0]*self.L)
        self.random_state = kwargs['state']
        self.layers = {}
        self.layers = [] # type: list[HAP_Layer]
        self.N = self.S.shape[0]
        self.H = np.zeros((self.N, self.N))
        self.H2 = np.zeros((self.N, self.N))
        self.verbose = kwargs.get('verbose', True)
        self.last_layer_rigid = kwargs.get('last_layer_rigid', False)
        self.pruning_thr = kwargs.get('pruning_thr', [0 for l in range(self.L)])


        #### The code is largely based on standard AP from sklearn ###
        # simple checks regarding damping value and shape of S
        ####CHECKS####
        if self.damping < 0.5 or self.damping >= 1:
            raise ValueError('damping must be >= 0.5 and < 1')
        
        if self.S.shape[0] != self.S.shape[1]:
            raise ValueError("S must be a square array (shape=%s)" % repr(self.S.shape))
        ##############
        self.ind = np.arange(self.N)

    def create_layers(self):
        '''
        Create layers of AP, that will be simultaneously updated 
        '''
        for i in range(self.L):
            self.layers.append(HAP_Layer(**{ # required params for every layer
                'container' : self,
                'pref'      : self.preferences[i],
                'p'         : self.p1s[i],
                'q'         : self.qs[i],
                'thr'       : self.thrs[i],
                'layer_number': i,
                'N_T'       : self.pruning_thr[i]
                })
            )
        return self

    def step(self): # do one message passing iteration on all layers
        for i in range(self.L): # compute messages from var to constraint nodes for all layers
            self.layers[i].Rs()
        for i in range(self.L): # compute messages from constraint to var nodes for all layers
            self.layers[i].As()

        if self.converged(): # check if converged
            print('converged')
            return True
        else:
            return False

    def prepare_results(self):
        '''
        called when all layers have converged
        '''
        for l in range(self.L): # prepare results individually for all layers
            self.layers[l].prepare_results(rigid=(self.last_layer_rigid and l == self.L - 1))

        #self.H2 is simply used to map all the edges to one matrix. Although we loose the distinction between
        #exemplars/edges of different layers, it helps us in capturing the global view of clusters for connected components calculation
        self.H2 = deepcopy(self.layers[0].decision.H_hat) # start from the lower most layer
        for l in range(0, self.L - 1): # move up the layers, startng from l=0
            for i in self.ind: # loop through all points
                if self.H2[i, i] == 1 and self.layers[l + 1].decision.H_hat[i, i] == 0: #if an exemplar i from lower layer is no longer an exemplar in upper layer, then:
                    children_of_i = np.where(self.layers[l].decision.H_hat[:, i] == 1)[0] # find children of i
                    clhs_of_i = np.where(self.layers[l + 1].decision.H_hat[i, :] == 1)[0] # find the point(s) that i has selected in upper layer
                    self.H2[children_of_i, i] = 0 # remove the assignments of children of i from i
                    for clh in clhs_of_i: # assign the children of i to parent(s) of i
                        self.H2[children_of_i, clh] = 1

    def converged(self): # check if converged or not. Simply loop through all layers. If all are converged, then it returns true.
        converged = True
        for i in range(self.L):
            if not self.layers[i].converged():
                converged = False
                break
        for i in range(self.L): # all layers converged, so let's save soft decisions. These will be thresholded afterwards to find H matrices for all layers
            self.layers[i].E = (self.layers[i].A +self.layers[i].R) > self.layers[i].thr

        return converged



    def run(self): # run the algo
        for it in range(self.max_itr):
            if self.verbose:
                print(it)
            converged = self.step() # move forward by one iteration and check if convergence is achieved or not
            if converged: # if converged, break out of loop
                print('converged after {} iterations'.format(it))
                break
            if it == self.max_itr - 1 and not converged: # if max iterations reached and function could not converge, then:
                print("could not converge")
        self.prepare_results()
        c_centers = np.unique(np.where(self.H2 == 1)[1]) # exemplars of H2 matrix
        c_labels = np.where(self.H2 == 1) # assignments data
        return c_centers, c_labels


class HAP_Layer:
    '''
    A sinlgle layer of HAP
    '''
    def __init__(self, **kwargs):
        self.container = kwargs['container'] #:Container
        self.pref = kwargs['pref']
        self.p = kwargs['p']
        self.q = kwargs['q']
        self.thr = kwargs['thr']
        self.N_T = kwargs['N_T']
        self.layer_number = kwargs['layer_number']
        self.current_itr = 0
        self.damping = self.container.damping
        self.agreed_iterations = 0
        self.S = deepcopy(self.container.S.astype(float))
        self.N = self.container.N
        self.H = np.zeros((self.N, self.N), dtype=bool)
        self.soft_H = np.zeros((self.N, self.N), dtype=float)
        self.decision = None # type: Decision
        if self.pref is None:
            self.pref = np.percentile(self.S, 70)


        # Initialize messages
        self.A = np.zeros((self.N, self.N))
        self.R = np.zeros((self.N, self.N))
        self.tau = np.zeros(self.N)
        self.Phi = np.zeros(self.N)
        self.tmp = np.zeros((self.N, self.N))
        self.tmpZZ = np.zeros((self.N, self.N))

        # add tiny noise to avoid degeneracy, just as in standard AP
        self.S += ((np.finfo(np.double).eps * self.S + np.finfo(np.double).tiny * 100) *
              self.container.random_state.randn(self.N, self.N))

        self.E_tmp = np.zeros(self.N)
        self.E = np.zeros((self.N, self.N))

        
    def Rs(self): # compute responsibilities
        # See equations 10 and 12 in http://www.psi.toronto.edu/publications/2011/HAP.pdf
        np.add(self.A, self.S, self.tmp)
        np.add(self.tmp, -np.maximum(0, self.tmp + self.q), self.tmp)
        I = np.argmax(self.tmp, axis=1)
        Y = self.tmp[self.container.ind, I]  # np.max(self.A + S, axis=1)
        self.tmp[self.container.ind, I] = -np.inf
        Y2 = np.max(self.tmp, axis=1)

        #########################################
        ## self.tmp = Rnew
        np.subtract(self.tmpZZ, Y[:, None], self.tmp)
        self.tmp[self.container.ind, I] = -Y2

        if self.prev() is not None:
            self.tmp = np.minimum(self.tau[:,None], self.tmp)
        np.add(self.tmp, self.S, self.tmp)

        # Damping
        self.tmp *= 1 - self.container.damping
        self.R *= self.container.damping
        self.R += self.tmp
        # if a lower layer exists, we need to prepare the message for that layer too
        if self.prev() is not None:
            self.prev().Phi *= self.prev().damping
            self.prev().Phi += (1 - self.prev().damping) * Y
        ############################################

    def As(self):
        # See equations 9 and 11 in http://www.psi.toronto.edu/publications/2011/HAP.pdf

        # self.tmp = Rp; compute availabilities
        np.maximum(self.R, 0, self.tmp)
        self.tmp.flat[::self.N + 1] = self.R.flat[::self.N + 1]

        # self.tmp = -Anew
        self.tmp -= np.sum(self.tmp, axis=0)
        if self.next() is not None:
            self.tmp -= (self.Phi+self.pref)[None,:]
        else:
            self.tmp -= self.pref
        dA = np.diag(self.tmp).copy()
        self.tmp.clip(0, np.inf, self.tmp)
        self.tmp.flat[::self.N + 1] = dA

        # Damping
        self.tmp *= 1 - self.container.damping
        self.A *= self.container.damping
        self.A -= self.tmp

        # If an upper layer exists, we also need to prepare message to that layer.
        if self.next() is not None:
            self.tmp = np.maximum(self.R, 0)
            self.tmp.flat[::self.N + 1] = self.R.flat[::self.N + 1]
            self.next().tau = self.next().damping * self.next().tau + \
                              (1 - self.next().damping)* (np.sum(self.tmp, axis=0)+self.pref)

        # Check for convergence
        E = (np.diag(self.A) + np.diag(self.R)) > self.thr #old
        ############################### new
        if np.array_equal(E, self.E_tmp):
            self.agreed_iterations += 1
        else:
            self.agreed_iterations = 0
            self.E_tmp = E

    def converged(self):
        return self.agreed_iterations >= self.container.conv_itr

    def next(self):
        if self.layer_number < self.container.L-1:
            return self.container.layers[self.layer_number + 1]
        else:
            return None

    def prev(self):
        if self.layer_number > 0:
            return self.container.layers[self.layer_number - 1]
        else:
            return None


    def prepare_results(self, rigid=False):
        # call decision process made in Decision class. Note that in SHAPE, rigid=false for lower layer and true for upper layer
        self.soft_H = self.A + self.R
        self.H = self.soft_H > self.thr
        self.decision = Decision(**{
            'H': self.H,
            'S': self.S,
            'N_T': self.N_T,
            'rigid': rigid
        }).compute()

#########################
