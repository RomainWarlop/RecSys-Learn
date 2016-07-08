import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
#from scipy.sparse.linalg import inv
#from scipy.linalg import sqrtm
from scipy.sparse.linalg import svds
from scipy.sparse import kron, diags, csr_matrix
from rslearn.graphReg.IterativeMethod import spJacobi
from rslearn.utils import sparse_matrix

def J(i,n):
    out = sparse.csr_matrix(([1],([i],[0])),shape=(n,1))
    return out

class graphALS(object):
    """
    Parameters
    ==========
    d : int
        Number of latent factors.
    num_users, num_items : int
        number of users and items to dimension the matrix.
    num_iters : int
        Number of iterations of alternating least squares.
    lbda : float
        Regularization constant.
    mu : float
        weight balance of the regularisation. For instance for the user : 
            lambda*(\sum_{user} n_u ||U_{user}||^2 + (1-mu)/mu*\sum_{l,q}||U_l-U_q||^2 W_{lq})
    seed : int
        if not None, fix the random seed
    reg: string
        what sould be the regularisation ? 
        - if "weighted" the regularisation 
        is weighted by the number of non empty value in each row and column
        - if None or "default" it will be classic l2 norm
        - other string will cause an error
    solve : string
        if 'exact' uses spsolve, if 'jacobi' uses Jacobi iterative method 
    verbose : bool
        Should it be verbose ?
    """
    
    def __init__(self,d,num_users,num_items,num_iters=20,lbda=0.1,mu=0.5, 
                 seed=None,reg="weighted",solve='exact',verbose=False):
        self.d = d
        self.num_users = num_users
        self.num_items = num_items
        self.num_iters = num_iters
        self.lbda = lbda
        self.mu = mu
        self.seed = seed
        self.reg = reg
        if self.reg is None:
            self.reg = "default"        
        self.solve = solve
        self.verbose = verbose
    
    def init_factors(self,method,U0=None,V0=None):
        if (U0 is not None) & (V0 is not None):
            self.U = U0
            self.V = V0
        elif method=="random":
            self.U = np.random.normal(size=(self.num_users,self.d))
            self.V = np.random.normal(size=(self.num_items,self.d))
        elif method=="svd":
                U,S,V = svds(self.train)
                self.U = np.dot(U,np.diag(S**.5))
                self.V = np.dot(np.diag(S**.5),V).T
    
    def fit(self,train,userGraph=None,itemGraph=None,method="random",U0=None,V0=None):
        """
        Learn factors from training set.
        User and item factors are fitted alternately.
        
        train : array-like of three columns 
            contains row index, column index, value of not null entries
        rowGraph : array-like of three columns
            contains the first two columns are the index of the linked rows,
            the third is the weight of the link
        colGraph : same as rowGraph for the column links
        method : string
            factor initialisation. Can be random, svd or given in U0 and V0
        U0, V0: array-like
            initial value for U and V. If not None, method is ignored
        """
        if self.seed is not None:
            np.random.seed(self.seed)
        
        self.userGraph = False
        self.itemGraph = False
        
        self.train = sparse_matrix(train,n = self.num_users, p = self.num_items)

        self.n_u = list(map(lambda u: self.train[u,:].nnz,range(self.num_users)))
        self.n_i = list(map(lambda i: self.train[:,i].nnz,range(self.num_items)))
        
        if userGraph is not None:
            self.userGraph = True
            self.A_user = sparse.csgraph.laplacian(sparse_matrix(userGraph,
                                     n = self.num_users, p = self.num_users,
                                     w = (1-self.mu)/self.mu))
            if self.reg=="weighted":
                self.A_user += sparse.diags(self.n_u)
            elif self.reg=="default":
                self.A_user += sparse.eye(self.num_users)
            #self.A_user_sqrt = kron(sqrtm(self.A_user.todense()).real,sparse.eye(self.d))
            self.A_user = kron(self.A_user,sparse.eye(self.d))
        else:
            if self.reg=="weighted":
                self.A_user = sparse.diags(self.n_u)
            elif self.reg=="default":
                self.A_user = sparse.eye(self.num_users)
            self.A_user = kron(self.A_user,sparse.eye(self.d))
                
        if itemGraph is not None:
            self.itemGraph = True
            self.A_item = sparse.csgraph.laplacian(sparse_matrix(itemGraph,
                                     n = self.num_items, p = self.num_items,
                                     w = (1-self.mu)/self.mu))
            if self.reg=="weighted":
                self.A_item += sparse.diags(self.n_i)
            elif self.reg=="default":
                self.A_item += sparse.eye(self.num_items)
            #self.A_item_sqrt = kron(sqrtm(self.A_item.todense()).real,sparse.eye(self.d))
            self.A_item = kron(self.A_item,sparse.eye(self.d))
        else:
            if self.reg=="weighted":
                self.A_item = sparse.diags(self.n_i)
            elif self.reg=="default":
                self.A_item = sparse.eye(self.num_items)
            self.A_item = kron(self.A_item,sparse.eye(self.d))
        
        self.U = np.random.normal(size=(self.num_users,self.d))
        self.V = np.random.normal(size=(self.num_items,self.d))

        for it in range(self.num_iters):
            if self.userGraph:
                self.U = self.user_iteration()
            else:
                for u in range(self.num_users):
                    indices = self.train[u].nonzero()[1]
                    if indices.size:
                        R_u = self.train[u,indices]
                        self.U[u,:] = self.update(indices,self.V,R_u.toarray().T)
                    else:
                        self.U[u,:] = np.zeros(self.d)

            if self.itemGraph:
                self.V = self.item_iteration()
            else:
                for i in range(self.num_items):
                    indices = self.train[:,i].nonzero()[0]
                    if indices.size:
                        R_i = self.train[indices,i]
                        self.V[i,:] = self.update(indices,self.U,R_i.toarray())
                    else:
                        self.V[i,:] = np.zeros(self.d)

            if self.verbose:
                print("end iteration "+str(it+1))

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
    
    def fitImplicit(self,train,alpha=10.,c="linear",eps=1E-8,
                    userGraph=None,itemGraph=None,method="random",
                    U0=None,V0=None):
        """
        Learn factors from training set with the implicit formula of Koren 
        "Collaborative Filtering for Implicit Feedback Datasets"
        User and item factors are fitted alternately.
        
        train : array-like of three columns 
            contains row index, column index, value of not null entries
        alpha : float
            Confidence weight
        c : string
            if c="linear", C = 1 + alpha*R
            if c="log", C = 1 + alpha*log(1 + R/eps)
        eps : float
            used only if c="log"
        rowGraph : array-like of three columns
            contains the first two columns are the index of the linked rows,
            the third is the weight of the link
        colGraph : same as rowGraph for the column links
        method : string
            factor initialisation. Can be random, svd or given in U0 and V0
        U0, V0: array-like
            initial value for U and V. If not None, method is ignored
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
        
        self.userGraph = False
        self.itemGraph = False
        
        self.train = sparse_matrix(train,n = self.num_users, p = self.num_items)

        self.n_u = list(map(lambda u: self.train[u,:].nnz,range(self.num_users)))
        self.n_i = list(map(lambda i: self.train[:,i].nnz,range(self.num_items)))
        
        if userGraph is not None:
            self.userGraph = True
            self.A_user = sparse.csgraph.laplacian(sparse_matrix(userGraph,
                                     n = self.num_users, p = self.num_users,
                                     w = (1-self.mu)/self.mu))
            if self.reg=="weighted":
                self.A_user += sparse.diags(self.n_u)
            elif self.reg=="default":
                self.A_user += sparse.eye(self.num_users)
            #self.A_user_sqrt = kron(sqrtm(self.A_user.todense()).real,sparse.eye(self.d))
            self.A_user = kron(self.A_user,sparse.eye(self.d))
                
        if itemGraph is not None:
            self.itemGraph = True
            self.A_item = sparse.csgraph.laplacian(sparse_matrix(itemGraph,
                                     n = self.num_items, p = self.num_items,
                                     w = (1-self.mu)/self.mu))
            if self.reg=="weighted":
                self.A_item += sparse.diags(self.n_i)
            elif self.reg=="default":
                self.A_item += sparse.eye(self.num_items)
            #self.A_item_sqrt = kron(sqrtm(self.A_item.todense()).real,sparse.eye(self.d))
            self.A_item = kron(self.A_item,sparse.eye(self.d))
        
        self.U = np.random.normal(size=(self.num_users,self.d))
        self.V = np.random.normal(size=(self.num_items,self.d))

        for it in range(self.num_iters):
            VV = self.V.T.dot(self.V)
            if self.userGraph:
                self.U = self.implicit_user_iteration(VV)
            else:
                for u in range(self.num_users):
                    indices = self.train[u].nonzero()[1]
                    if indices.size:
                        R_u = self.train[u,indices]
                        self.U[u,:] = self.implicit_update(indices,self.V,VV,R_u.toarray()[0])
                    else:
                        self.U[u,:] = np.zeros(self.d)
            
            UU = self.U.T.dot(self.U)
            if self.itemGraph:
                self.V = self.implicit_item_iteration(UU)
            else:
                for i in range(self.num_items):
                    indices = self.train[:,i].nonzero()[0]
                    if indices.size:
                        R_i = self.train[indices,i]
                        self.V[i,:] = self.implicit_update(indices,self.U,UU,R_i.toarray().T[0])
                    else:
                        self.V[i,:] = np.zeros(self.d)

            if self.verbose:
                print("end iteration "+str(it+1))


    def user_iteration(self):
        block = []
        b = sparse.lil_matrix((self.num_users*self.d,1))
        for u in range(self.num_users):
            if self.n_u[u]:
                indices = self.train[u,:].nonzero()[1]
                V = self.V[indices,:].reshape(int(self.n_u[u]),self.d)
                VVT = V.T.dot(V)
                block.extend([VVT])
                b[self.d*u:(u+1)*self.d,:] = V.T.dot(self.train[u,indices].toarray().T) # check here
        M = sparse.block_diag(tuple(block))
        
        if self.solve=='exact':
            solve_vecs = spsolve(self.lbda*self.A_user + M, b)
        elif self.solve=='jacobi':
            solve_vecs = spJacobi(self.lbda*self.A_user + M, b,max_iter=5)
            solve_vecs = np.array(solve_vecs).reshape(len(solve_vecs))
        solve_vecs = solve_vecs.reshape((self.num_users,self.d))
        return solve_vecs


    def implicit_user_iteration(self,HH):
        block = []
        b = sparse.lil_matrix((self.num_users*self.d,1))
        for u in range(self.num_users):
            if self.n_u[u]:
                indices = self.train[u,:].nonzero()[1]
                Hix = csr_matrix(self.V[indices,:])
                R = self.train[u,indices].toarray()[0]
                C = diags(self.transfo(R,intercept=False),shape=(len(R),len(R)))
                block.extend([HH + Hix.T.dot(C).dot(Hix) + np.diag(self.lbda*len(R)*np.ones(self.d))])
                C = diags(self.transfo(R),shape=(len(R),len(R)))
                b[self.d*u:(u+1)*self.d,:] = (C.dot(Hix)).sum(axis=0).T
        M = sparse.block_diag(tuple(block))
        
        if self.solve=='exact':
            solve_vecs = spsolve(self.lbda*self.A_user + M, b)
        elif self.solve=='jacobi':
            solve_vecs = spJacobi(self.lbda*self.A_user + M, b,max_iter=5)
            solve_vecs = np.array(solve_vecs).reshape(len(solve_vecs))
        solve_vecs = solve_vecs.reshape((self.num_users,self.d))
        return solve_vecs


    def item_iteration(self):
        block = []
        b = sparse.lil_matrix((self.num_items*self.d,1))
        for i in range(self.num_items):
            if self.n_i[i]>0:
                indices = self.train[:,i].nonzero()[0]
                U = self.U[indices,:].reshape(int(self.n_i[i]),self.d)
                UUT = U.T.dot(U)
                block.extend([UUT])
                b[self.d*i:(i+1)*self.d,:] = U.T.dot(self.train[indices,i].toarray())
        M = sparse.block_diag(tuple(block))
        
        if self.solve=='exact':
            solve_vecs = spsolve(self.lbda*self.A_item + M, b)
        elif self.solve=='jacobi':
            solve_vecs = spJacobi(self.lbda*self.A_item + M, b,max_iter=5)
            solve_vecs = np.array(solve_vecs).reshape(len(solve_vecs))
        solve_vecs = solve_vecs.reshape((self.num_items,self.d))
        return solve_vecs


    def implicit_item_iteration(self,HH):
        block = []
        b = sparse.lil_matrix((self.num_items*self.d,1))
        for i in range(self.num_items):
            if self.n_i[i]>0:
                indices = self.train[:,i].nonzero()[0]
                Hix = csr_matrix(self.U[indices,:])
                R = self.train[indices,i].toarray().T[0]
                C = diags(self.transfo(R,intercept=False),shape=(len(R),len(R)))
                block.extend([HH + Hix.T.dot(C).dot(Hix) + np.diag(self.lbda*len(R)*np.ones(self.d))])
                C = diags(self.transfo(R),shape=(len(R),len(R)))
                b[self.d*i:(i+1)*self.d,:] = (C.dot(Hix)).sum(axis=0).T
        M = sparse.block_diag(tuple(block))
        
        if self.solve=='exact':
            solve_vecs = spsolve(self.lbda*self.A_item + M, b)
        elif self.solve=='jacobi':
            solve_vecs = spJacobi(self.lbda*self.A_item + M, b,max_iter=5)
            solve_vecs = np.array(solve_vecs).reshape(len(solve_vecs))
        solve_vecs = solve_vecs.reshape((self.num_items,self.d))
        return solve_vecs

  
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
        return np.dot(np.linalg.inv(M),Hix.T.dot(R)).reshape(self.d)
    
    
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


#==============================================================================
#******************************************************************************
#==============================================================================

class multipleGraphALS(object):
    """
    Parameters
    ==========
    d : int
        Number of latent factors.
    num_users, num_items : int
        number of users and items to dimension the matrix.
    lbda : float
        Regularization constant.
    num_iters : int
        Number of iterations of alternating least squares.
    reg: string
        what sould be the regularisation ? 
        - if "weighted" the regularisation 
        is weighted by the number of non empty value in each row and column
        - if None or "default" it will be classic l2 norm
        - other string will cause an error
    solve : string
        if 'exact' uses spsolve, if 'jacobi' uses Jacobi iterative method 
    verbose : bool
        Should it be verbose ?
    """
    
    def __init__(self,d,num_users,num_items,num_iters=20,lbda=0.1,
                 reg=None,solve='exact',verbose=False):
        self.lbda = lbda
        self.d = d
        self.num_iters = num_iters
        self.num_users = num_users
        self.num_items = num_items
        self.verbose = verbose
        self.solve = solve
        self.reg = reg
        if self.reg is None:
            self.reg = "default"
    
    def fit(self,train,graphs=None):
        """
        Learn factors from training set.
        User and item factors are fitted alternately.
        
        train : array-like of three columns 
            contains row index, column index, value of not null entries
        graphsList : dict of tuple. 
            Each tuple is two array-like of three columns
            The first element of each tuple correspond to the user graph
                The first two columns are the index of the linked rows,
                the third is the weight of the link
            The second element of each tuple correspond to the item graph
                The first two columns are the index of the linked columns,
                the third is the weight of the link
        U0, V0 : array-like
            initialization of the decomposition. If None, initiate with random values
        
        The key of the dict can be anything but it will be the key in the graph output, so
        be sure that user and item's graphs keys are different
        
        Example :
        if graphs has 2 key-value of key 'a' and 'b'
        Then fit will return in self.graph a dictionnary where 
        self.graph['a'] will contains the results with a user graph 
        defined by graphs['a'][0] and item graph defined by graphs['a'][1]
        self.graph['b'] will contains the results with a user graph 
        defined by graphs['a'][0] and item graph defined by graphs['b'][1]
        """
        
        self.graph = {}
        if graphs is not None:
            for key in graphs.keys():
                if self.verbose:
                    print('learn user graph',key)
                self.graph[key] = graphALS(d=self.d, 
                                        num_users = self.num_users,
                                        num_items = self.num_items,
                                        num_iters = self.num_iters,
                                        lbda = self.lbda,
                                        reg = self.reg,
                                        solve = self.solve,
                                        verbose = self.verbose)
                self.graph[key].fit(train,userGraph=graphs[key][0],
                                          itemGraph=graphs[key][1])

    def fitImplicit(self,train,alpha=10.,c="linear",eps=1E-8,
                    graphs=None):
        """
        Learn factors from training set with the implicit formula of Koren 
        "Collaborative Filtering for Implicit Feedback Datasets"
        User and item factors are fitted alternately.
        
        train : array-like of three columns 
            contains row index, column index, value of not null entries
        alpha : float
            Confidence weight
        c : string
            if c="linear", C = 1 + alpha*R
            if c="log", C = 1 + alpha*log(1 + R/eps)
        eps : float
            used only if c="log"
        graphsList : dict of tuple. 
            Each tuple is two array-like of three columns
            The first element of each tuple correspond to the user graph
                The first two columns are the index of the linked rows,
                the third is the weight of the link
            The second element of each tuple correspond to the item graph
                The first two columns are the index of the linked columns,
                the third is the weight of the link
        U0, V0 : array-like
            initialization of the decomposition. If None, initiate with random values
        
        The key of the dicts can be anything but it will be the key in the graph output, so
        be sure that user and item's graphs keys are different
        
        Example :
        if userGraphs has 2 key-value of key 'a' and 'b' and itemGraphs has
        1 key-value of key 'c'
        Then fit will return in self.graph a dictionnary where 
        self.graph['a'] will contains the results for userGraph['a'] and 
        itemGraph None
        self.graph['b'] will contains the results for userGraph['b'] and 
        itemGraph None
        self.graph['c'] will contains the results for userGraph None and 
        itemGraph['c']
        """
        self.alpha = alpha
        self.c = c
        self.eps = eps
        
        self.graph = {}
        if graphs is not None:
            for key in graphs.keys():
                if self.verbose:
                    print('learn user graph',key)
                self.graph[key] = graphALS(d=self.d, 
                                        num_users = self.num_users,
                                        num_items = self.num_items,
                                        num_iters = self.num_iters,
                                        lbda = self.lbda,
                                        reg = self.reg,
                                        solve = self.solve,
                                        verbose = self.verbose)
                self.graph[key].fitImplicit(train,userGraph=graphs[key][0],
                                            alpha=self.alpha,
                                            c=self.c,eps=self.eps,
                                            itemGraph=graphs[key][1])

#==============================================================================
#******************************************************************************
#==============================================================================

# adapt regularization
class graphMF(object):
    """
    Graph Regularized Non-negative Matrix Factorization for Data Representation
    Deng Cai, Member, IEEE, Xiaofei He, Senior Member, IEEE, Jiawei Han, Fellow, IEEE
    Thomas S. Huang, Life Fellow, IEEE
    
    Parameters
    ==========
    d : int
        the rank constraint
    num_rows, num_cols : int
        number of rows and columns in the matrix
    num_iters : int
        number of iterations
    lbda : float
        regularization constant
    lrate : float
        learning rate
    verbose : bool
        should it be verbose ?
    """
    def __init__(self,d,num_rows,num_cols,num_iters=20,lbda=0.1,lrate=0.001,
                 reg=None,verbose=False):
        self.lbda = lbda
        self.d = d
        self.lrate = lrate
        self.num_iters = num_iters
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.verbose = verbose
        self.reg = reg
        if self.reg is None:
            self.reg = "default"
    
    def fit(self,train,rowGraph=None,colGraph=None,U0=None,V0=None):
        """
        train : array-like of three columns 
            contains row index, column index, value of not null entries
        rowGraph : array-like of three columns
            contains the first two columns are the index of the linked rows,
            the third is the weight of the link
        colGraph : same as rowGraph for the column links
        U0, V0 : array-like
            initialization of the decomposition. If None, initiate with random values
        """
        ind = train.nonzero()
        
        self.U = U0
        self.V = V0
        
        if self.U is None:
            self.U = np.random.random_sample((self.num_rows,self.d))
        if self.V is None:
            self.V = np.random.random_sample((self.num_cols,self.d))
        
        n_u = list(map(lambda u: train[u,:].nnz,range(self.num_rows)))
        n_i = list(map(lambda i: train[:,i].nnz,range(self.num_cols)))
        
        if rowGraph is not None:
            W_u = sparse.csr_matrix(
                        (rowGraph[2], (rowGraph[0], rowGraph[1])),
                        shape=(self.num_rows,self.num_rows))
            D_u = np.array(W_u.sum(axis=0))[0]
            D_u += n_u
        else:
            W_u = sparse.lil_matrix((self.num_rows,self.num_rows))
            D_u = n_u
        
        if colGraph is not None:
            W_i = sparse.csr_matrix(
                        (colGraph[2], (colGraph[0], colGraph[1])),
                        shape=(self.num_cols,self.num_cols))
            D_i = np.array(W_i.sum(axis=0))[0]
            D_i += n_i
        else:
            W_i = sparse.lil_matrix((self.num_cols,self.num_cols))
            D_i = n_i

        for it in range(self.num_iters):
            for a in range(len(ind[0])):
                i = ind[0][a]
                j = ind[1][a]
                e = train[i,j]-self.U[i,:].dot(self.V[j,:].T)
                
                sub_ind_i = W_u[i,:].nonzero()[1]
                sub_ind_j = W_i[:,j].nonzero()[0]
                self.U[i,:] += self.lrate*(e*self.V[j,:]-self.lbda*(D_u[i]*self.U[i,:]+W_u[i,sub_ind_i].toarray().dot(self.U[sub_ind_i,:]).reshape(self.d)))
                self.V[j,:] += self.lrate*(e*self.U[i,:]-self.lbda*(D_i[j]*self.V[j,:]+W_i[sub_ind_j,j].toarray().T.dot(self.V[sub_ind_j,:]).reshape(self.d)))
            
            if self.verbose:
                print("end iteration "+str(it+1))


#==============================================================================
#******************************************************************************
#==============================================================================

class GNMF(object):
    """
    Graph Regularized Non-negative Matrix Factorization for Data Representation
    Deng Cai, Member, IEEE, Xiaofei He, Senior Member, IEEE, Jiawei Han, Fellow, IEEE
    Thomas S. Huang, Life Fellow, IEEE
    
    Parameters
    ==========
    d : int
        the rank constraint
    num_rows, num_cols : int
        number of rows and columns in the matrix
    num_iters : int
        number of iterations
    lbda : float
        regularization constant
    seed: int
        if not None, fix the random seed
    verbose : bool
        should it be verbose ?
    """
    def __init__(self,d,num_rows,num_cols,num_iters=20,lbda=0.1,seed=None,verbose=False):
        """
        ALS with graph regularization with multiple graph learnt independently 
        for both user and item. 
        d : the rank constraint
        lbda : regularization paramter
        num_iterations : number of iterations
        verbose : should it be verbose
        reg: default <- graph regularized
        """
        self.d = d
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_iters = num_iters
        self.lbda = lbda
        self.seed = seed
        self.verbose = verbose
    
    def fit(self,train,rowGraph=None,colGraph=None,reg="default",U0=None,V0=None):
        
        if self.seed is not None:
            np.random.seed(self.seed)
        
        num_users,num_items = train.shape
        if "matrix" not in str(type(train)):
            train = sparse_matrix(train,n = self.num_rows, p = self.num_cols)
        
        self.U = U0
        self.V = V0
        
        if self.U is None:
            self.U = np.random.random_sample((self.num_rows,self.d))
        if self.V is None:
            self.V = np.random.random_sample((self.num_cols,self.d))
        
        n_u = list(map(lambda u: train[u,:].nnz,range(self.num_rows)))
        n_i = list(map(lambda i: train[:,i].nnz,range(self.num_cols)))
        
        if rowGraph is not None:
            W_u = sparse_matrix(rowGraph,n = self.num_rows, p = self.num_rows)
            D_u= W_u.sum(axis=0)
            if reg=="weighted":
                 D_u += sparse.diags(n_u)
        else:
            W_u = sparse.lil_matrix((self.num_rows,self.num_rows))
            D_u= W_u.sum(axis=0)
            if reg=="weigthed":
                D_u += sparse.diags(n_u)
        
        if colGraph is not None:
            W_i = sparse_matrix(colGraph,n = self.num_cols, p = self.num_cols)
            D_i = W_i.sum(axis=0)
            if reg=="weigthed":
                D_i += sparse.diags(n_i)
        else:
            W_i = sparse.lil_matrix((self.num_cols,self.num_cols))
            D_i = sparse.diags(n_i)
            if reg=="weigthed":
                D_i += sparse.diags(n_i)

        for it in range(self.num_iters):
            self.U *= (train.dot(self.V)+self.lbda*W_u.dot(self.U))/(self.U.dot(self.V.T).dot(self.V)+self.lbda*D_u.dot(self.U))
            self.V *= (train.T.dot(self.U)+self.lbda*W_i.dot(self.V))/(self.V.dot(self.U.T).dot(self.U)+self.lbda*D_i.dot(self.V))
            self.U = np.nan_to_num(self.U)
            self.V = np.nan_to_num(self.V)
            
            if self.verbose:
                print("end iteration "+str(it+1))
        
    def prediction(self):
        return np.argmax(self.U,axis=1)
        
        
        
        
        