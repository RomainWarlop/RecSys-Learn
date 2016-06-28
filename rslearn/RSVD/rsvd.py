"""
RSVD as proposed by Paterek with the improved version
"""

import numpy as np

#==========================================================
# RSVD and improved RSVD algorithms 
#==========================================================

class RSVD(object):
    """
    Parameters
    ==========
    d : int
        Number of latent factors.
    lbda : float
        Regularization constant for user x item features.
    lbda2 : float
        Regularization constant for bias.
    num_iters : int
        Number of iterations of alternating least squares.
    improved : bool
        if True then user and item bias are added.
    """

    def __init__(self,d,lbda=0.02,lbda2=0.05,lrate=0.001,num_iters=20,improved=False,verbose=False):
        self.d = d
        self.lbda = lbda
        self.lbda2 = lbda2
        self.lrate = lrate
        self.num_iters = num_iters
        self.verbose = verbose
        self.improved = improved # is this the improved RSVD code or regular RSVD
   
    def init_factors(self,num_factors):
        return self.d**-0.5*np.random.random_sample((num_factors,self.d))

    def fit(self,train,U0=None,V0=None,batch_size = 1):
        """
        Learn factors from training set. User and item factors are
        fitted alternately.
        Parameters
        ==========
        train : scipy.sparse.csr_matrix User-item matrix.
        item_features : array_like, shape = [num_items, num_features]
            Features for each item in the dataset, ignored here.
        """

        num_users,num_items = train.shape
        ind = train.nonzero()

        self.U = U0
        self.V = V0
        if self.U is None:
            self.U = self.init_factors(num_users)
        if self.V is None:
            self.V = self.init_factors(num_items)

        if self.improved:
            self.global_mean = train[ind].mean()
            self.C = np.zeros(num_users)
            self.D = np.zeros(num_items)
        
        for it in np.arange(self.num_iters):
            if self.verbose:
                print('iteration',it)
            # fit user factors

            for a in range(len(ind[0])):
                i = ind[0][a]
                j = ind[1][a]
                
                if self.improved:
                    e = train[i,j]-self.U[i,:].dot(self.V[j,:].T)-self.C[i]-self.D[j]
                else:
                    e = train[i,j]-self.U[i,:].dot(self.V[j,:].T)
                
                self.U[i,:] += self.lrate*(e*self.V[j,:]-self.lbda*self.U[i,:])
                self.V[j,:] += self.lrate*(e*self.U[i,:]-self.lbda*self.V[j,:])
                if self.improved:
                    self.C[i] += self.lrate*(e - self.lbda2*(self.C[i] + self.D[j] - self.global_mean))
                    self.D[j] += self.lrate*(e - self.lbda2*(self.C[i] + self.D[j] - self.global_mean))
