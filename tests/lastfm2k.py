import pandas as pd
import numpy as np
import json
from time import time
import itertools
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm
from sklearn.neighbors import kneighbors_graph
from rslearn.utils import sparseMatrix, rankMeasure, getLine_fromdict
from rslearn.ALSWR.alswr import ALS
from rslearn.graphReg.graph_regularized import graphALS

import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)

path = '/home/romain/Documents/BitBucket/PhD/RecSys-Learn/'
json_data=open(path+"tests/lfm2k_parameters.json").read()
params = json.loads(json_data)
K = 5 # number of cross validation

# input
data = pd.read_csv(path+'data/lastfm2k/user_artists.dat',sep="\t")
data.columns = ['user','artist','weight']

# filter out low active artist
countsItem = data['artist'].value_counts()
selectedArtist = countsItem.index[countsItem>10]
artistFrequent = pd.DataFrame({'artist': selectedArtist})
data = pd.merge(data,artistFrequent)

# filter out low active user
countsUser = data['user'].value_counts()
selectedUser= countsUser.index[countsUser>10]
userFrequent = pd.DataFrame({'user': selectedUser})
data = pd.merge(data,userFrequent)

userPivot = pd.DataFrame({'user': list(set(data['user'])),
                          'userId': np.arange(len(set(data['user'])))})

artistPivot = pd.DataFrame({'artist': list(set(data['artist'])),
                          'artistId': np.arange(len(set(data['artist'])))})

data = pd.merge(pd.merge(data,userPivot),artistPivot)

del data['user']
del data['artist']
data = data[['userId','artistId','weight']]

numUser = len(set(data['userId']))
numItem = len(set(data['artistId']))

np.random.seed(0)
data['cv'] = np.random.randint(0,K,data.shape[0])

perf = pd.read_csv(path+'tests/perf_lfm2k.csv')

#==============================================================================
# ALS-WR
#==============================================================================
if params['ALS-WR']['learn']=="True":
    print("Learn ALS-WR")
    #regs = params['ALS-WR']['params']['reg']
    #lambdas = params['ALS-WR']['params']['lambda']
    #rank = params['ALS-WR']['params']['rank']
    #alphas = params['ALS-WR']['params']['alpha']
    #epss = params['ALS-WR']['params']['eps']
    regs = ["default"]
    lambdas = [0.8]
    rank = [20]
    alphas = [10]
    epss= [10]
    
    current_perf = dict.fromkeys(['model','reg','lambda','rank','alpha','eps','crossval'])
    for elt in itertools.product(*[regs,lambdas,rank,alphas,epss,[0]]):
        reg, l, r, alpha, eps, k = elt
        current_perf.update({'model':'ALS-WR','reg':reg,'lambda':l,
                            'rank':r,'alpha':alpha,'eps':eps,
                            'crossval':k})
        print(current_perf)
        bob = ALS(d=r,num_users=numUser,num_items=numItem,lbda=l,seed=0,
                  reg=reg,verbose=True)
        t0 = time()
        bob.fitImplicit(data[data['cv']!=k],alpha=alpha,c="log",eps=eps)
        T = time()-t0
        Rhat = bob.U.dot(bob.V.T)
        R_test = sparseMatrix(data,k,include=True,names=list(data.columns)[:3])
        rank = rankMeasure(R_test,Rhat)
        print(rank)
        ind = getLine_fromdict(perf,current_perf)
        perf.loc[ind] = ['ALS-WR',reg,l,r,alpha,eps,k,rank,T]
    print('-'*50)

#==============================================================================
# graphALS
#==============================================================================

# Graph Construction
userGraph = pd.read_csv(path+'data/lastfm2k/user_friends.dat',sep="\t")
userGraph.columns = ['user','friend']

userGraph = pd.merge(userGraph,userPivot)
userGraph = pd.merge(userGraph,userPivot,left_on='friend',right_on='user')
userGraph = userGraph[['userId_x','userId_y']]
userGraph.columns = ['userId','friendId']
userGraph['binary_weight'] = 1.0

userMatrixGraph = csr_matrix((userGraph['binary_weight'], 
                        (userGraph['userId'],userGraph['friendId'])), 
                        shape=(numUser,numUser))

# change weight to the number of common friend
userGraph['dot_product_weight'] = 0.0
userGraph['heat_kernel_weight'] = 0.0
for (u,f) in zip(userGraph['userId'],userGraph['friendId']):
    if u<f:
        F_uf = (userGraph['userId']==u) & (userGraph['friendId']==f)
        F_fu = (userGraph['userId']==f) & (userGraph['friendId']==u)
        dot = userMatrixGraph[u].dot(userMatrixGraph[f].T)[0,0]
        userGraph.loc[F_uf,'dot_product_weight'] = dot+1
        userGraph.loc[F_uf,'dot_product_weight'] = dot+1
        
        #heat = np.e**-(norm(userMatrixGraph[u]-userMatrixGraph[f])**2)
        #userGraph.loc[F_uf,'heat_kernel_weight'] = heat
        #userGraph.loc[F_uf,'heat_kernel_weight'] = heat

neighborsGraph = kneighbors_graph(userMatrixGraph,n_neighbors=20,mode='distance')
userNeighborsGraph = pd.DataFrame({'userId':neighborsGraph.nonzero()[0],
                                   'neighborId':neighborsGraph.nonzero()[1],
                                   'weight':np.array(neighborsGraph[neighborsGraph.nonzero()])[0]})


if params['graphALS']['learn']=="True":
    print("Learn graphALS")
    #regs = params['graphALS']['params']['reg']
    #lambdas = params['graphALS']['params']['lambda']
    #rank = params['graphALS']['params']['rank']
    #alphas = params['ALS-WR']['params']['alpha']
    #epss = params['ALS-WR']['params']['eps']
    regs = ["weighted"]
    lambdas = [0.8]
    rank = [10]
    alphas = [10]
    epss= [100]
    mus = [0.5]
    
    current_perf = dict.fromkeys(['model','reg','lambda','rank','alpha','eps','mu','crossval'])
    for elt in itertools.product(*[regs,lambdas,rank,alphas,epss,mus,[0]]):
        reg, l, r, alpha, eps, mu, k = elt
        current_perf.update({'model':'graphALS','reg':reg,'lambda':l,
                            'rank':r,'alpha':alpha,'eps':eps,'mu':mu,
                            'crossval':k})
        print(current_perf)
        bob = graphALS(d=r,num_users=numUser,num_items=numItem,lbda=l,mu=mu,
                       seed=0,reg=reg,verbose=True)
        t0 = time()
        bob.fitImplicit(train = data[data['cv']!=k],
                        alpha=alpha,
                        c="log",
                        eps=eps,
                        userGraph = userGraph[['userId','friendId','dot_product_weight']],
                        itemGraph = None,
                        method="svd")
        T = time()-t0
        Rhat = bob.U.dot(bob.V.T)
        R_test = sparseMatrix(data,k,include=True,names=list(data.columns)[:3])
        rank = rankMeasure(R_test,Rhat)
        print(rank)
        ind = getLine_fromdict(perf,current_perf)
        perf.loc[ind] = ['graphALS',reg,l,r,alpha,eps,k,rank,T]
    print('-'*50)

perf.to_csv(path+'tests/perf_lfm2k.csv',index=False)

perf.loc[perf.groupby('model')['rankMeasure'].idxmin()]
