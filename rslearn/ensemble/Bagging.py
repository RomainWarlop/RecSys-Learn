import numpy as np
from rslearn.utils import sparse_matrix_input, sparse_matrix
from rslearn.ALSWR.alswr import ALS

class bag_ALS(object):
    """
    Parameters
    ==========
    d : int
        Number of latent factors.
    num_users, num_items : int
        number of users and items to dimension the matrix.
    num_iters : int
        Number of iterations of alternating least squares.
    num_estimators : int
        Number of estimators to boostrap
    lbda : float
        Regularization constant
    dropping : string
        dropping method for boostraping. Can be :
        - uniform : each entry will be pull or not uniformly at random. The importance of 
            an item will then be the l2 norm on the missing item point
        - item : each item will be either entirely pull from the dataset or 
            entirely not pull            
    sampling : float < 1 
        the sampling parameter definining the number of entries / items to pull
    seed : int
        if not None, fix the random seed
    verbose : bool
        Should it be verbose ?
    """
    def __init__(self,d,num_users,num_items,num_iters=20,num_estimators=10,lbda=0.8,
                 dropping='uniform',sampling=0.8,seed=None,verbose=False):
        self.d = d
        self.num_users = num_users
        self.num_items = num_items
        self.num_iters = num_iters
        self.num_estimators = num_estimators
        self.lbda = lbda
        self.dropping = dropping
        self.sampling = sampling
        self.seed = seed
        self.verbose = verbose
           
    def fit(self,train,U0=None,V0=None):
        if self.seed is not None:
            np.random.seed(self.seed)
            
        self.train = sparse_matrix_input(train)
        self.f = {}
        self.clf = {}
        
        if self.dropping=='item':
            # create a train and a test set from the train set
            sample = np.random.random(len(self.train))
            self.train_train = self.train[sample<=0.9,:]
            self.train_test = self.train[sample>0.9,:]
        
        for est in range(self.num_estimators):
            if self.verbose:
                print("estimator",est+1)
            if self.dropping=='uniform':
                self.f[est] = np.random.random(len(self.train))
                R = self.train[self.f[est]<self.sampling,:]
            elif self.dropping=='item':
                self.f[est] = np.random.choice(range(self.num_items),
                                            int(self.sampling*self.num_items),
                                            replace=False)
                R = self.train_train[[i for i in range(len(self.train_train)) if self.train_train[i,1] in self.f[est]],:]
            self.clf[est] = ALS(d = self.d, 
                      num_users = self.num_users,
                      num_items = self.num_items,
                      lbda = self.lbda, 
                      num_iters= self.num_iters,
                      verbose = self.verbose)
            self.clf[est].fit(train = R, U0 = U0, V0 = V0)
    
    def predict(self):
        Rhats = {}
        Rhat = np.zeros(shape=(self.num_users,self.num_items))
        for est in range(self.num_estimators):
            Rhats[est] = self.clf[est].U.dot(self.clf[est].V.T)
            Rhat += Rhats[est]/self.num_estimators
        
        return Rhats, Rhat
    
    def importance(self,Rhats = None):
        if Rhats is None:
            Rhats, _ = self.predict()
        
        item_imp = np.zeros(self.num_items)
    
        if self.dropping=='uniform':
            drop_count = np.zeros(self.num_items)
            for est in range(self.num_estimators):
                mask = self.train
                mask[:,2] = 1.
                W_out = sparse_matrix(mask[self.f[est]>=self.sampling,:],
                                  self.num_users,self.num_items)
                R_test = sparse_matrix(self.train[self.f[est]>=self.sampling,:],
                                  self.num_users,self.num_items)
                
                # compute the error at the unselected point for each item
                tmp = (R_test-W_out.multiply(Rhats[est])).power(2).sum(0)
                tmp = np.array(tmp)[0]
                den = W_out.sum(0)
                den = np.array(den)[0]
                drop_count[den>0]+=1
                den[den==0] = 1
                tmp /= den
                tmp = np.array(np.sqrt(tmp))
                item_imp += tmp
            drop_count[drop_count==0] = 1
            item_imp = item_imp/drop_count
        elif self.dropping=='item':
            drop_count = np.zeros(self.num_items)
            mask = self.train_test
            mask[:,2] = 1.
            for est in range(self.num_estimators):
                # filter out the unselected item from the train_test set
                tmp = [i for i in range(len(self.train_test)) if self.train_test[i,1] in self.f[est]]
                W_out = sparse_matrix(mask[tmp,:],self.num_users,self.num_items)
                R_test = sparse_matrix(self.train_test[tmp,:],
                                  self.num_users,self.num_items)
                tmp = ((np.array(R_test[W_out.nonzero()])[0]-Rhats[est][W_out.nonzero()])**2).sum()
                den = W_out.nnz
                tmp = np.sqrt(tmp/den)
                not_selected = [i for i in range(self.num_items) if i not in self.f[est]]
                drop_count[not_selected] += 1
                item_imp[not_selected] += tmp
            drop_count[drop_count==0] = 1
            drop_count[drop_count==0] = 1
            item_imp = item_imp/drop_count
        return item_imp