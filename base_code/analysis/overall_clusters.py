from enum import Enum
import numpy as np
# import matplotlib.pyplot as plt


class ChainAggregatorMethod(Enum):
    # Choose how to find connected components. All three give same result.
    SYMMETRIC_H = 1 # Convert H to symmetric form and then find connected components of resulting graph
    ASYMMETRIC_H = 2 # Find connected components of a graph with directed edges.
    SYMMETRIC_H_RECURSION = 3 # Find connected components throuh recursion on graph, built using symmetrized H matrix



class ChainAggregator:
    '''
    Class responsible for implementing ChainAggregatorMethod.
    '''
    def __init__(self, H:np.ndarray, method:ChainAggregatorMethod=ChainAggregatorMethod.SYMMETRIC_H):
        self.method = method
        self.H = H

    def get_clusters(self):
        if self.method == ChainAggregatorMethod.SYMMETRIC_H_RECURSION:
            clusters = RecursiveChainAggregator(H=self.H).aggregate()
        elif self.method == ChainAggregatorMethod.SYMMETRIC_H:
            clusters = SymmetricHAggregator(H=self.H).aggregate()
        elif self.method == ChainAggregatorMethod.ASYMMETRIC_H:
            clusters = AsymmetricHAggregator(H=self.H).aggregate()
        else:
            raise ValueError('unknown method passed for aggregation')

        return clusters



class AggregatorBaseClass:
    '''
    Parent class for functions related to finding connected components. 
    It summarizes the basic functionality of an Aggregator class
    '''
    def __init__(self, H):
        self.H = H
        self.n = self.H.shape[0]
        self.H2 = np.bitwise_or(self.H, self.H.T)
        self.visited = np.array([False] * self.n)
        self.clusters = {}

    def refine_num_clusters(self):
        '''
        implemented in child classes
        '''
        pass

    def aggregate(self):
        '''
        Simply calls the aggregate method
        :return:  dictionary of clusters
        '''
        self.refine_num_clusters()
        return self.clusters

class RecursiveChainAggregator(AggregatorBaseClass):
    '''
    Used when ChainAggregatorMethod.SYMMETRIC_H_RECURSION is selected
    '''
    def __init__(self, H):
        super().__init__(H)


    def refine_num_clusters(self):
        for u in range(self.n): # loop over all points
            if not self.visited[u]: # check whether a point u visited or not. If not, then start a new cluster from it
                self.visited[u] = True # add u to the list of visited points
                k = len(self.clusters.keys()) # k will give us the key for the new cluster
                self.clusters[k] = [u] # u is the first member of the new cluster.
                self.unify_clusters(u, k) # recursively find the points connected to u.

    def unify_clusters(self, u, k):
        connections = np.where(self.H2[u,:]==True)[0] #find points connected to u
        for v in connections: # loop through every connected point v
            if not self.visited[v]: # if v is not visited yet, mark it as visited
                self.visited[v] = True
                self.clusters[k].append(v) # add it to the cluster of u
                self.unify_clusters(v, k) # find points further connected to v, to add them into the cluster of u


class SymmetricHAggregator(AggregatorBaseClass):
    '''
    Used when ChainAggregatorMethod.SYMMETRIC_H is selected
    '''
    def __init__(self, H:np.ndarray):
        super().__init__(H)

    def refine_num_clusters(self):
        for i in range(self.n): # loop through all points
            if not self.visited[i]: # proceed only if i is not visited yet
                self.visited[i] = True # add i to the list of visited points
                k = len(self.clusters.keys()) # generate the key for new cluster
                self.clusters[k] = [i] # add i as the first element of this new cluster
                neighbours = np.where(self.H2[:, i])[0] # find neighbours of i
                super_edge = [v for v in neighbours if not self.visited[v]] # super_edge consists of the neighbours, not yet visited
                while not super_edge == []: # if super edge is empty, it means all neighbours of i have been already visited. So, we can move to next point. Otherwise, we proceed
                    self.visited[super_edge] = True # add all these neighbours to visited list
                    self.clusters[k].extend(super_edge) # add all these neighbours to the cluster containing i
                    neighbours = np.where(np.bitwise_or.reduce(self.H2[:, super_edge], axis=1))[0] # find neoghbours of all the points in super_edge
                    super_edge = [v for v in neighbours if not self.visited[v]] # find the new super_edge, that consists of only non-visited elements of this new neighborhood


class AsymmetricHAggregator(AggregatorBaseClass):
    '''
    Used when ChainAggregatorMethod.ASYMMETRIC_H is selected 
    '''
    def __init__(self, H:np.ndarray):
        super().__init__(H)

    def refine_num_clusters(self):
        exemplars = np.where(self.H.diagonal())[0] # find the list of exemplars
        for i in range(len(exemplars)): # loop through exemplars
            if not self.visited[i]: # if i is not yet visited, proceed
                self.visited[i] = True # add i to list of visited points.
                k = len(self.clusters.keys()) # find key for new cluster
                self.clusters[k] = [i] # include i as thr first element fo this new cluster
                children = np.where(self.H[:, i])[0] # find points that have selected i as their exemplar
                parents = np.where(self.H[i, :])[0] # find points that i has selected as its exemplars
                neighbours = np.array(list(set(parents).union(set(children)))) # group children and parents into a list called neighbours
                super_edge = [v for v in neighbours if not self.visited[v]] # find all connections of these neighbours, and filter the ones not yet visited.
                while not super_edge == []: # if super_edge is empty, we have found one connected component. If not, we need to add more points.
                    self.visited[super_edge] = True # mark all points in super_edge list as visited
                    self.clusters[k].extend(super_edge)
                    children = np.where(np.bitwise_or.reduce(self.H[:, super_edge], axis=1))[0] # find children of super_edge
                    parents = np.where(np.bitwise_or.reduce(self.H[super_edge, :], axis=0))[0] # find parents of super_edge
                    neighbours = np.array(list(set(parents).union(set(children)))) # join children and parents into a list of neighbours
                    super_edge = [v for v in neighbours if not self.visited[v]] # create a new super edge by filtering only the non-visited neighbours






