import numpy as np
from scipy.sparse import diags, csr_matrix
from rslearn.utils import sparse_matrix

class neighborhood_svdpp(object):
    
    def __init__(self,num_users,num_items,lbda=0.8,num_iters=20,n_neigh=10,
                 seed=None,verbose=False):
        self.num_users = num_users
        self.num_items = num_items
        self.lbda = lbda
        self.num_iters = num_iters
        self.n_neigh = n_neigh
        self.seed = seed
        self.verbose = verbose
    
    def fit











