"""
Alternating Least Square with lambda weighted regularization
Zhou et al
both explicit and implicit vrsion (of Koren) are implemented
code can be run in parallel in local with setting parallel = True
"""
import numpy as np
from multiprocessing import Pool
from scipy.sparse import diags, csr_matrix
from rslearn.utils import sparse_matrix

def linear_transfo(R,alpha,intercept=True):
    if intercept:
        return 1+alpha*R
    else:
        return alpha*R

def log_transfo(R,alpha,eps,intercept=True):
    if intercept:
        return 1+alpha*np.log(1+R/eps)
    else:
        return alpha*np.log(1+R/eps)

class ALS(object):
    """
    Parameters
    ==========
    d : int
        Number of latent factors.
    num_users, num_items : int
        number of users and items to dimension the matrix.
    lbda : float
        Regularization constant.
    lbda2 : float
        Regularization constant for improved model (not available for implicit for now).
    num_iters : int
        Number of iterations of alternating least squares.
    parallel : bool
        Should update be done in parallel (in local)
    seed : int
        if not None, fix the random seed
    reg: string
        what sould be the regularisation ? 
        - if "weighted" the regularisation 
        is weighted by the number of non empty value in each row and column
        - if None or "default" it will be classic l2 norm
    improved : bool
        Should a user and a item bias be computed ?
    verbose : bool
        Should it be verbose ?
    """

    def __init__(self,d,num_users,num_items,lbda=0.8,lbda2=0.8,num_iters=20,
                 parallel=False,seed=None,reg="weighted",improved=False,verbose=False):
        self.d = d
        self.num_users = num_users
        self.num_items = num_items
        self.lbda = lbda
        self.lbda2 = lbda2
        self.num_iters = num_iters
        self.parallel = parallel
        self.seed = seed
        self.reg = reg
        if self.reg is None:
            self.reg = "default"
        self.improved = improved
        self.verbose = verbose
   
    def init_factors(self,num_factors,assign_values=True):
        if assign_values:
            return self.d**-0.5*np.random.random_sample((num_factors,self.d))
        return np.empty((num_factors,self.d))
            
    def fit(self,train,U0=None,V0=None):
        """
        Learn factors from training set. User and item factors are
        fitted alternately.
        
        Parameters
        ==========
        train : scipy.sparse.csr_matrix 
            User-item matrix.
        U0, V0 : array-like
            initialization of the decomposition. If None, initiate with random values
        """
        if self.seed is not None:
            np.random.seed(self.seed)
        
        train = sparse_matrix(train,n = self.num_users, p = self.num_items)
        
        self.U = U0
        self.V = V0
        
        if self.improved:
            self.C = np.random.random_sample(self.num_users)
            self.D = np.random.random_sample(self.num_items)
            
        if self.U is None:
            #self.U = self.init_factors(self.num_users,False)
            self.U = np.random.normal(size=(self.num_users,self.d))
        if self.V is None:
            #self.V = self.init_factors(self.num_items)
            self.V = np.random.normal(size=(self.num_items,self.d))
        for it in np.arange(self.num_iters):
            
            if self.parallel:
                pool = Pool()
                res = [pool.apply_async(self.parallel_update, (u,train,True)) for u in range(self.num_users)]
                self.U = np.array([r.get() for r in res])
                
                pool = Pool()
                res = [pool.apply_async(self.parallel_update, (i,train,False)) for i in range(self.num_items)]
                self.V = np.array([r.get() for r in res])
            else:
                for u in range(self.num_users):
                    indices = train[u].nonzero()[1]
                    if indices.size:
                        R_u = train[u,indices]
                        if self.improved:
                            self.C[u] = self.bias_update(True,u,indices,R_u.toarray().T)
                            self.U[u,:] = self.user_imp_update(u,indices,R_u.toarray())
                        else:
                            self.U[u,:] = self.update(indices,self.V,R_u.toarray().T)
                    else:
                        self.U[u,:] = np.zeros(self.d)
                        if self.improved:
                            self.C[u] = 0

                for i in range(self.num_items):
                    indices = train[:,i].nonzero()[0]
                    if indices.size:
                        R_i = train[indices,i]
                        if self.improved:
                            self.D[i] = self.bias_update(False,i,indices,R_i.toarray().T)
                            self.V[i,:] = self.item_imp_update(i,indices,R_i.toarray())
                        else:
                            self.V[i,:] = self.update(indices,self.U,R_i.toarray().T[0])
                    else:
                        self.V[i,:] = np.zeros(self.d)
                        if self.improved:
                            self.D[i] = 0
                        
            if self.verbose:
                print('end iteration',it+1)
    
    def linear_transfo(self,R,intercept=True):
        if intercept:
            return 1+self.alpha*R
        else:
            return self.alpha*R
    
    def log_transfo(self,R,intercept=True):
        if intercept:
            return 1+self.alpha*np.log(1+R/self.eps)
        else:
            return self.alpha*np.log(1+R/self.eps)

    def fitImplicit(self,train,alpha=10.,c="linear",eps=1E-8,U0=None,V0=None):  
        """
        Learn factors from training set with the implicit formula of Koren 
        "Collaborative Filtering for Implicit Feedback Datasets"
        User and item factors are fitted alternately.
        
        Parameters
        ==========
        train : scipy.sparse.csr_matrix 
            User-item matrix.
        alpha : float
            Confidence weight, confidence c = 1 + alpha*r where r is the observed "rating".
        c : string
            if c="linear", C = 1 + alpha*R
            if c="log", C = 1 + alpha*log(1 + R/eps)
        eps : float
            used only if c="log"
        U0, V0 : array-like
            initialization of the decomposition. If None, initiate with random values
        """
        if self.seed is not None:
            np.random.seed(self.seed)
        
        # we define a global fonction transfo that is either linear_transfo
        # or log_transfo. This prevent to make the if else check for each 
        # user and for each item at each iteration !
        if c=="linear":
            self.transfo = self.linear_transfo
        elif c=="log":
            self.transfo = self.log_transfo
        
        self.alpha = alpha
        self.c = c
        self.eps = eps
        
        train = sparse_matrix(train,n = self.num_users, p = self.num_items)
        
        self.U = U0
        self.V = V0
            
        if self.U is None:
            #self.U = self.init_factors(self.num_users,False)
            self.U = np.random.normal(size=(self.num_users,self.d))
        if self.V is None:
            #self.V = self.init_factors(self.num_items)
            self.V = np.random.normal(size=(self.num_items,self.d))
        for it in np.arange(self.num_iters):
            
            if self.parallel:
                VV = self.V.T.dot(self.V)
                pool = Pool()
                res = [pool.apply_async(self.parallel_implicit_update, (u,VV,train,True)) for u in range(self.num_users)]
                self.U = np.array([r.get() for r in res])
                
                UU = self.U.T.dot(self.U)
                pool = Pool()
                res = [pool.apply_async(self.parallel_implicit_update, (i,UU,train,False)) for i in range(self.num_items)]
                self.V = np.array([r.get() for r in res])
            else:
                VV = self.V.T.dot(self.V)
                for u in range(self.num_users):
                    # get (positive i.e. non-zero scored) items for user
                    indices = train[u].nonzero()[1]
                    if indices.size:
                        R_u = train[u,indices]
                        self.U[u,:] = self.implicit_update(indices,self.V,VV,R_u.toarray()[0])
                    else:
                        self.U[u,:] = np.zeros(self.d)
    
                UU = self.U.T.dot(self.U)
                for i in range(self.num_items):
                    indices = train[:,i].nonzero()[0]
                    if indices.size:
                        R_i = train[indices,i]
                        self.V[i,:] = self.implicit_update(indices,self.U,UU,R_i.toarray().T[0])
                    else:
                        self.V[i,:] = np.zeros(self.d)
            
            if self.verbose:
                print('end iteration',it+1)
    
    
    def parallel_update(self,ind,train,user=True):
        if user:
            indices = train[ind].nonzero()[1]
            Hix = self.V[indices,:]
            R_u = train[ind,indices]
            R = R_u.toarray().T
        else:
            indices = train[:,ind].nonzero()[0]
            Hix = self.U[indices,:]
            R_i = train[indices,ind]
            R = R_i.toarray().T[0]
        if len(indices)>0:
            HH = Hix.T.dot(Hix)
            if self.reg=="weighted":
                M = HH + np.diag(self.lbda*len(R)*np.ones(self.d))
            elif self.reg=="default":
                M = HH + np.diag(self.lbda*np.ones(self.d))
            return np.linalg.solve(M,Hix.T.dot(R)).reshape(self.d)
        else:
            return np.zeros(self.d)
        
    def update(self,indices,H,R):
        """
        Update latent factors for a single user or item.
        """
        Hix = H[indices,:]
        HH = Hix.T.dot(Hix)
        if self.reg=="weighted":
            M = HH + np.diag(self.lbda*len(R)*np.ones(self.d))
        elif self.reg=="default":
            M = HH + np.diag(self.lbda*np.ones(self.d))
        return np.linalg.solve(M,Hix.T.dot(R)).reshape(self.d)
    
    def parallel_implicit_update(self,ind,HH,train,user=True):
        if user:
            indices = train[ind].nonzero()[1]
            Hix = self.V[indices,:]
            R_u = train[ind,indices]
            R = R_u.toarray().T
        else:
            indices = train[:,ind].nonzero()[0]
            Hix = self.U[indices,:]
            R_i = train[indices,ind]
            R = R_i.toarray().T[0]
        if len(indices)>0:
            C = diags(self.transfo(R,intercept=False),shape=(len(R),len(R)))
            if self.reg=="weighted":
                M = HH + Hix.T.dot(C).dot(Hix) + np.diag(self.lbda*len(R)*np.ones(self.d))
            elif self.reg=="default":
                M = HH + Hix.T.dot(C).dot(Hix) + np.diag(self.lbda*np.ones(self.d))
            C = diags(self.transfo(R),shape=(len(R),len(R)))
            return np.linalg.solve(M,(C.dot(Hix)).sum(axis=0).T).reshape(self.d)
        else:
            return np.zeros(self.d)
    
    def implicit_update(self,indices,H,HH,R):
        """
        Implicit update latent factors for a single user or item.
        """
        # manque R entre Hix.T.dot(Hix)
        Hix = csr_matrix(H[indices,:])
        C = diags(self.transfo(R,intercept=False),shape=(len(R),len(R)))
        if self.reg=="weighted":
            M = HH + Hix.T.dot(C).dot(Hix) + np.diag(self.lbda*len(R)*np.ones(self.d))
        elif self.reg=="default":
            M = HH + Hix.T.dot(C).dot(Hix) + np.diag(self.lbda*np.ones(self.d))
        C = diags(self.transfo(R),shape=(len(R),len(R)))
        return np.linalg.solve(M,(C.dot(Hix)).sum(axis=0).T).reshape(self.d)
    
    def user_imp_update(self,u,indices,R):
        """
        Update latent factors for a single user or item.
        """
        Hix = self.V[indices,:]
        HH = Hix.T.dot(Hix)
        if self.reg=="weighted":
            M = HH + np.diag(self.lbda*len(R)*np.ones(self.d))
        elif self.reg=="default":
            M = HH + np.diag(self.lbda*np.ones(self.d))
        return np.linalg.solve(M,Hix.T.dot(R[0]-self.D[indices]-self.C[u])).reshape(self.d)
    
    def item_imp_update(self,i,indices,R):
        """
        Update latent factors for a single user or item.
        """
        Hix = self.U[indices,:]
        HH = Hix.T.dot(Hix)
        if self.reg=="weighted":
            M = HH + np.diag(self.lbda*len(R)*np.ones(self.d))
        elif self.reg=="default":
            M = HH + np.diag(self.lbda*np.ones(self.d))
        return np.dot(np.linalg.inv(M),Hix.T.dot(R[0]-self.C[indices]-self.D[i])).reshape(self.d)
    
    def bias_update(self,user,u,indices,R):
        if user:
            if self.reg=="weighted":
                return (1.+self.lbda2*len(R))/(R-self.U[u,:].dot(self.V[indices,:].T)-self.D[indices]).sum()
            elif self.reg=="default":
                return (1.+self.lbda2)/(R-self.U[u,:].dot(self.V[indices,:].T)-self.D[indices]).sum()
        else:
            if self.reg=="weighted":
                return (1.+self.lbda2*len(R))/(R-self.V[indices,:].dot(self.V[u,:].T)-self.C[indices]).sum()
            elif self.reg=="default":
                return (1.+self.lbda2)/(R-self.V[indices,:].dot(self.V[u,:].T)-self.C[indices]).sum()