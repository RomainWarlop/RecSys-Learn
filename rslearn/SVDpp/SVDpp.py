import numpy as np
import pandas as pd
from scipy.stats import pearsonr

class neighborhood_svdpp(object):
    
    def __init__(self,num_users,num_items,n_neigh=10,num_iters=20,lbda=0.8,
                 gamma=0.005,seed=None,verbose=False):
        self.num_users = num_users
        self.num_items = num_items
        if n_neigh=='inf':
            self.n_neigh = num_items
        else:
            self.n_neigh = n_neigh
        self.num_iters = num_iters
        self.lbda = lbda
        self.gamma = gamma
        self.seed = seed
        self.verbose = verbose
    
    def fit(self,train):
        
        if self.seed is not None:
            np.random.seed(self.seed)
        
        # First step : fit the b_ui : b_ui = mu + bc_u + bc_i
        tab = pd.DataFrame(train)
        self.mu = tab['val'].mean()
        self.bc_u = np.random.normal(size=self.num_users)
        self.bc_i = np.random.normal(size=self.num_items)
        for it in range(self.num_iters):
            for ind in range(len(train['row'])):
                u = train['row'][ind]
                i = train['col'][ind]
                r = train['val'][ind]
                e = r-self.mu-self.bc_u[u]-self.bc_i[i]
                self.bc_u[u] += self.gamma*(e-self.lbda*self.bc_u[u])
                self.bc_i[i] += self.gamma*(e-self.lbda*self.bc_i[i])

        # Second step : compute similarity score between items
        if self.n_neigh<self.num_items:
            S = np.zeros((self.num_items,self.num_items))
            for i in range(self.num_items):
                for j in range(i+1,self.num_items):
                    u_i = set(tab.loc[tab['col']==i,'row'])
                    u_j = set(tab.loc[tab['col']==j,'row'])
                    u_ij = list(u_i & u_j)
                    n_ij = len(u_ij)
                    if n_ij>0:
                        subtab = tab[tab['col']==i]
                        r_i = subtab.loc[[k for k in subtab.index if subtab['row'].loc[k] in u_ij],['val','row']]
                        r_i = list(r_i.sort_values('row')['val'])
                        subtab = tab[tab['col']==j]
                        r_j = subtab.loc[[k for k in subtab.index if subtab['row'].loc[k] in u_ij],['val','row']]
                        r_j = list(r_j.sort_values('row')['val'])
                        rho = (n_ij)/(n_ij+100)*pearsonr(r_i,r_j)[0]
                        S[i,j] = rho
                        S[j,i] = rho
                        
        # Third step : fit the other parameters
        self.b_u = np.random.normal(size=self.num_users)
        self.b_i = np.random.normal(size=self.num_items)
        self.w = np.random.normal(size=self.num_items**2).reshape(self.num_items,self.num_items)
        self.c = np.random.normal(size=self.num_items**2).reshape(self.num_items,self.num_items)

        for it in range(self.num_iters):
            for ind in len(train['row']):
                u,i,r = tab.loc[ind,['row','col','val']]
                e = r-self.mu-self.b_u[u]-self.b_i[i]
                self.b_u[u] += self.gamma*(e-self.lbda*self.b_u[u])
                self.b_i[i] += self.gamma*(e-self.lbda*self.b_i[i])
                if self.n_neigh<self.num_items:
                    tmp = pd.DataFrame([S[i,:],np.arange(self.num_items)],columns=['cor','item'])
                    R_kiu = tmp.sort_values('cor')['val'][:self.n_neigh]
                    R_u = set(tab.loc[tab['row']==u,'row'])
                    R_kiu = list(set(R_kiu) & R_u)
                else:
                    R_kiu = np.arange(self.num_items)
                if len(R_kiu)>0:
                    coeff = len(R_kiu)**-0.5
                    for j in R_kiu:
                        self.w[i,j] += self.gamma*(coeff*e*(tab.loc[(tab['row']==u) & (tab['col']==i),'val']-self.mu-self.bc_u[u]-self.bc_i[j])-self.lbda*self.w[i,j])
                        self.c[i,j] += self.gamma*(coeff*e-self.lbda*self.c[i,j])
            if self.verbose:
                print("end iteration",it+1)









