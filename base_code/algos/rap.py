from sklearn.utils import as_float_array
from base_code.algos.decision import Decision
import numpy as np


class Log:
    def __init__(self, verbose):
        self.verbose = verbose

    def print(self, msg):
        if self.verbose:
            print(msg)

class EAP:
    """
    It is copied from scikit learn, with a small change of \eta, as mentioned in the paper.
    This is a hybrid version with both SCAP and EAP implemented togther.
    Moreover, the decision process is changed as per the paper.
    """
    def __init__(self, **kwargs):
        """
        :param preference: preference
        :param q: penalty. Default=-np.inf
        :param eps: epsilon to define \partial_j. See paper. Default=0
        :param S: NxN Similarity Matrix
        :param max_iter: Maximum allowed iterations. Default=2000
        :param damping: Damping Factor for messages. Default=0.8
        :param convergence_iter: convergence iterations. Default=200
        :param thr: decision threshold. Default=0
        :param p: SCAP penalty for multiple exemplars selection. Usually kept to -np.inf for EAP
        :param random_state: Numpy RandomState. Default = np.random.RandomState(0)
        
        Example:        
            >>> af = EAP(**{
                    'S': S,
                    'preference': pref,
                    'q': q,
                    'eps': eps,
                    'N_T': 2
                    }).run()
            final_H = af.decision.H_hat.copy()
            
        """
        self.kwargs = kwargs
        self.log = Log(self.kwargs.get('verbose', True))
        self.c_itr = 0
        self.converged = False
        self.decision = None # type: Decision

    def run(self):
        p = self.kwargs.get('p', np.inf)
        q = self.kwargs.get('q', -np.inf)
        eps = self.kwargs.get('eps', 0) # epsilon of G(.) function
        thr = self.kwargs.get('thr', 0) # in case if you want to simply lower the threshold for decision of h_ij as 0/1
        S = self.kwargs['S'] # similarity matrix
        convergence_iter = self.kwargs.get('convergence_iter', 200)
        max_iter = self.kwargs.get('max_iter', 2000) # maxmum iterations, after which the algo gives result even if not converged
        damping = self.kwargs.get('damping', 0.8)
        preference = self.kwargs.get('preference', None)
        random_state = self.kwargs.get('random_state', np.random.RandomState(0)) # to keep consistency in results


        #############################
        ### This code is largely based upon standard AP code of sklearn ###
        S = as_float_array(S, copy=True)
        n_samples = S.shape[0]

        if S.shape[0] != S.shape[1]:
            raise ValueError("S must be a square array (shape=%s)" % repr(S.shape))

        if preference is None: # when preference is not given
            preference = np.percentile(S, 70)
        if damping < 0.5 or damping >= 1:
            raise ValueError('damping must be >= 0.5 and < 1')
        # print(preference)


        c_itr = 0

        #############################################
        # Now we make a matrix where h_ij = 1 if j falls in eps neighborhood of i
        connections = []

        for indx in range(n_samples):
            connections.append(np.array(np.where(S[indx, :] > eps)).squeeze())

        for indx in range(n_samples):
            try:
                len(connections[indx])
            except: # in case of only one neighbour
                connections[indx] = [connections[indx]]

        mask = np.zeros((n_samples, n_samples), dtype='bool') #mask matrix, whose entries will refer to neighbours
        for indx in range(n_samples):
            mask[indx, connections[indx]] = True

        mask = mask.T
        S.flat[::(n_samples + 1)] = preference # preference assigned to diagonal

        # Initialize messages
        A = np.zeros((n_samples, n_samples))
        R = np.zeros((n_samples, n_samples))
        Psi = np.zeros((n_samples, n_samples))
        F = np.zeros((n_samples)) # sum of messages from G(.) constraints. See paper

        # Intermediate results
        tmp = np.zeros((n_samples, n_samples))

        # Remove degeneracies, same as in standard AP
        S += ((np.finfo(np.double).eps * S + np.finfo(np.double).tiny * 100) *
              random_state.randn(n_samples, n_samples))

        # Execute parallel EAP/SCAP updates
        E_tmp = np.zeros((n_samples))
        ind = np.arange(n_samples)
        tmpZZ = np.zeros((n_samples, n_samples))

        for it in range(max_iter):
            if it % 50 == 0: # log after 50 iterations, if logging is enabled
                self.log.print(it)

            # Beta
            np.add(S, A, tmp) # A+S
            tmp.flat[::n_samples + 1] += F # add F to diagonal of A+S matrix. So, \beta_{ij} = (A+S)_{ij} + 1(i==j)F_i
            Beta = tmp.copy()

            #This portion computes the messages \eta_ij
            tmp = Beta - np.maximum(0, Beta + q)
            I = np.argmax(tmp, axis=1)
            Y = tmp[ind, I]
            tmp[ind, I] = -np.inf
            Y2 = np.max(tmp, axis=1)
            tmp = tmpZZ - Y[:, None]
            tmp[ind, I] = -Y2
            Eta = tmp.copy()

            # R: \rho_{ij} = (S+\eta)_{ij} + 1(i==j)F_i
            np.add(S, Eta, tmp)
            tmp.flat[::n_samples + 1] += F
            dA = np.diag(tmp).copy()
            tmp.flat[::n_samples + 1] = np.maximum(-p, dA) # this portion relates to SCAP equations. For p=inf, this is ignored. See SCAP equations for reference: https://academic.oup.com/bioinformatics/article/23/20/2708/230273
            R = tmp.copy()

            # Phi
            np.add(S, A + Eta, tmp)
            tmpN = tmp.diagonal() + F # tmpN_{ii} = s_{ii} + a_{ii}+\eta_{ii} + F_i
            tmp = tmpN[:, None] - Psi
            Phi = tmp.copy()

            # Psi
            tmp = Phi.copy()
            tmp[~mask] = -np.inf # remove non-neighbours from competition of argmax
            I = np.argmax(tmp, axis=0) #find argmax
            Y = tmp[I, ind] # save max values
            tmp[I, ind] = -np.inf # remove current argmax to find second argmax indices
            Y2 = np.max(tmp, axis=0) # find second best indices
            tmp = tmpZZ + Y # tmpZZ is an NxN matrix of zeros. So, we now have Y mapped to an NxN matrix
            tmp[I, ind] = Y2 # the indices of 1st argmax are assigned 2nd argmax values
            tmp = -np.maximum(0, tmp) # clipping at 0
            tmp[~mask] = 0 # we only need messages for points in neighborhood. So rest are deleted.

            Psi = damping * Psi + (1 - damping) * tmp
            F = np.sum(Psi, axis=1) # sum of all \psi_{ij} s.t. i \in \partial_j
            #########################################
            #########################################

            # A: availability messages. Same as standard AP if we ignore SCAP
            np.maximum(R, 0, tmp)
            tmp.flat[::n_samples + 1] = R.flat[::n_samples + 1]

            # tmp = Anew
            tmp -= np.sum(tmp, axis=0)
            tmp = -tmp
            dA = np.diag(tmp).copy()

            dA = np.minimum(p, dA)# if this part is ignored(eg when p=inf) A has same form as standard AP. For p not equal to inf, A has the form given in https://academic.oup.com/bioinformatics/article/23/20/2708/230273
            tmp.clip(-np.inf, 0, tmp)

            tmp.flat[::n_samples + 1] = dA

            # Damping
            tmp *= 1 - damping
            A *= damping
            A += tmp
            ##########################
            ############################
            E = np.diagonal(A + R) > thr
            if np.array_equal(E, E_tmp):
                c_itr += 1
            else:
                c_itr = 0
                E_tmp = E

            if c_itr == convergence_iter:
                self.log.print("converged after %d iterations" % it)
                self.c_itr = it
                self.converged = True
                break
            elif it == max_iter - 1:
                self.log.print("could not converge")
            else:
                pass


        self.soft_messages = A + R
        self.H = self.soft_messages > thr
        self.decision = Decision(**{
            'H' : self.H,
            'S' : self.kwargs['S'],
            'N_T' : self.kwargs.get('N_T', 0)
        }).compute()

        return self
