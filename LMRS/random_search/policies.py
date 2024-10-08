'''
Copied from https://raw.githubusercontent.com/modestyachts/ARS/master/code/policies.py
'''


import numpy as np
from random_search.filter import get_filter

class Policy(object):

    def __init__(self, policy_params):

        self.ob_dim = policy_params['ob_dim']
        self.ac_dim = policy_params['ac_dim']
        self.weights = np.empty(0)

        # a filter for updating statistics of the observations and normalizing inputs to the policies
        self.observation_filter = get_filter(policy_params['ob_filter'], shape = (self.ob_dim,))
        self.update_filter = True
        
    def update_weights(self, new_weights):
        self.weights[:] = new_weights[:]
        return

    def get_weights(self):
        return self.weights

    def get_observation_filter(self):
        return self.observation_filter

    def act(self, ob):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError

class LinearPolicy(Policy):
    """
    Linear policy class that computes action as <w, ob>. 
    """

    def __init__(self, policy_params):
        Policy.__init__(self, policy_params)
        self.weights = np.zeros((self.ac_dim, self.ob_dim), dtype = np.float64)

    def act(self, ob):
        ob = self.observation_filter(ob, update=self.update_filter)
        return np.dot(self.weights, ob)

    def get_stats_only(self):
        mu, std = self.observation_filter.get_stats()
        aux = np.asarray([mu, std])
        return aux

    def get_weights_plus_stats(self):
        # this is where the problem exists
        mu, std = self.observation_filter.get_stats()
        # mu_expanded = mu[np.newaxis, :]  # Shape becomes (1, 17)
        # std_expanded = std[np.newaxis, :]  # Shape becomes (1, 17)

        # Concatenate weights, mu, and std along the first axis (rows)
        aux = (self.weights,mu,std)
        # aux = np.concatenate([self.weights, mu_expanded, std_expanded], axis=0)
        # aux = np.asarray([self.weights, mu, std])
        # in the shape of 8*17s
        return aux
 
