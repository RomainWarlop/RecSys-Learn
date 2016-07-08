# coding: utf-8

# # Test on the movielens 100k dataset 
# ** this small dataset help to check if everything goes well in each algorithms and parameters **

import pandas as pd
import numpy as np
import json
from time import time
from sklearn.metrics import mean_squared_error
from copy import deepcopy
import itertools

#from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import RandomForestClassifier as RFC

from rslearn.utils import sparse_matrix, RMSE, getLine_fromdict, extract_year
from rslearn.ALSWR.alswr import ALS
from rslearn.RSVD.rsvd import RSVD
from rslearn.graphReg.graph_regularized import graphALS, multipleGraphALS, GNMF
from rslearn.ensemble.Bagging import bag_ALS

path = '/home/romain/Documents/BitBucket/PhD/RecSys-Learn/'
json_data=open(path+"tests/ml100k_parameters.json").read()
params = json.loads(json_data)
K = 5 # number of cross validation
data = pd.read_csv(path+'data/movielens100k/ml100k.data',sep="\t",index_col=False,names=['user','item','rating','timestamp'])
del data['timestamp']

np.random.seed(0)
data['cv'] = np.random.randint(0,K,data.shape[0])

userContext = pd.read_csv(path+"data/movielens100k/u.user",sep="|",index_col=False,
                       names=['user','age','gender','occupation','zipcode'])
userContext['user'] -= 1

itemContext = pd.read_csv(path+"data/movielens100k/u.item",sep="|",index_col=False, encoding = "ISO-8859-1",
                   names=['item','movietitle','releasedate',
                   'videoreleasedate','IMDbURL','unknown','Action',
                   'Adventure','Animation','Children','Comedy','Crime',
                   'Documentary','Drama','Fantasy','FilmNoir','Horror',
                   'Musical','Mystery','Romance','SciFi','Thriller','War',
                   'Western'])
itemContext['item'] -= 1

genres = ['Action','Adventure','Animation','Children','Comedy','Crime',
          'Documentary','Drama','Fantasy','FilmNoir','Horror','Musical',
          'Mystery','Romance','SciFi','Thriller','War','Western']

perf = pd.read_csv(path+'tests/perf_ml100k.csv')

numUser = len(set(data['user']))
numItem = len(set(data['item']))

data['user'] -= 1
data['item'] -= 1

data = data.rename(columns={'user':'row','item':'col','rating':'val'})


#==============================================================================
# ALS-WR
#==============================================================================
if params['ALS-WR']['learn']=="True":
    print("Learn ALS-WR")
    lambdas = params['ALS-WR']['params']['lambda']
    rank = params['ALS-WR']['params']['rank']
    
    current_perf = dict.fromkeys(['model','lambda','rank','crossval'])
    for elt in itertools.product(*[lambdas,rank,range(K)]):
        l, r, k = elt
        current_perf['model'] = 'ALS-WR'
        current_perf['lambda'] = l
        current_perf['rank'] = r
        current_perf['crossval'] = k
        print(current_perf)
        bob = ALS(d=r,num_users=numUser,num_items=numItem,lbda=l,parallel=True)
        train = data[data['cv']!=k][['row','col','val']].to_dict(orient='list')
        test = data[data['cv']==k][['row','col','val']].to_dict(orient='list')
        t0 = time()
        bob.fit(train)
        T = time()-t0
        Rhat = bob.U.dot(bob.V.T)
        R_test = sparse_matrix(test,numUser,numItem)
        rmse = RMSE(R_test,Rhat)
        print(rmse)
        ind = getLine_fromdict(perf,current_perf)
        perf.loc[ind,['model','rank','lambda','crossval','rmse','runningTime']] = ['ALS-WR',r,l,k,rmse,T]
    print('-'*50)



#==============================================================================
# bagging ALS-WR
#==============================================================================
if params['bagging ALS-WR']['learn']=="True":
    print("Learn bagging ALS-WR")
    lambdas = params['bagging ALS-WR']['params']['lambda']
    rank = params['bagging ALS-WR']['params']['rank']
    
    current_perf = dict.fromkeys(['model','lambda','rank','crossval'])
    for elt in itertools.product(*[lambdas,rank,range(K)]):
        l, r, k = elt
        current_perf['model'] = 'bagALS-WR'
        current_perf['lambda'] = l
        current_perf['rank'] = r
        current_perf['crossval'] = k
        print(current_perf)
        bob = bag_ALS(d=r,num_users=numUser,num_items=numItem,split='item',sampling=.9,
                      num_estimators = 10,lbda=l,verbose=True)
        train = data[data['cv']!=k][['row','col','val']].to_dict(orient='list')
        test = data[data['cv']==k][['row','col','val']].to_dict(orient='list')
        t0 = time()
        bob.fit(train)
        T = time()-t0
        Rhats, Rhat = bob.predict()
        R_test = sparse_matrix(test,numUser,numItem)
        rmse = RMSE(R_test,Rhat)
        print(rmse)
        ind = getLine_fromdict(perf,current_perf)
        perf.loc[ind,['model','rank','lambda','crossval','rmse','runningTime']] = ['bagALS-WR',r,l,k,rmse,T]
    print('-'*50)


##################################
# MODIFY FROM HERE
##################################
##################################
##################################
##################################
##################################
##################################
##################################
##################################
##################################
##################################


#==============================================================================
# contextual ALS-WR
#==============================================================================
if params['contextual ALS-WR']['learn']=="True":
    print("Learn contextual ALS-WR")
    lambdas = params['contextual ALS-WR']['params']['lambda']
    rank = params['contextual ALS-WR']['params']['rank']
    
    data_context = deepcopy(data)
    #data_context['rating'] /= 5
    #==========================================================================
    # User Context 
    #==========================================================================

    # occupation
    userContext['occupation_fact'], _ = pd.factorize(userContext['occupation'])
    occupation = userContext[['user','occupation_fact']]
    occupation.is_copy = False
    occupation.columns = ['user','item']
    occupation['item'] += numItem
    occupation['rating'] = data['rating'].mean()
    occupation['cv'] = -1
    data_context = pd.concat([data_context,occupation])
    
    # age
    userContext['age_slice'] = pd.cut(userContext['age'],range(0,100,5))
    userContext['age_slice_fact'], _ = pd.factorize(userContext['age_slice'])
    age = userContext[['user','age_slice_fact']]
    age.is_copy = False
    age.columns = ['user','item']
    age['item'] += numItem + len(set(userContext['occupation']))
    age['rating'] = data['rating'].mean()
    age['cv'] = -1
    data_context = pd.concat([data_context,age])
    
    #==========================================================================
    # Item Context 
    #==========================================================================
    
    # genre
    g = 0
    for genre in genres:
        tmp = itemContext['item'][itemContext[genre]==1]
        tmp = pd.DataFrame({'user':numUser+g,'item':tmp,'rating':data['rating'].mean(),'cv':-1})
        data_context = pd.concat([data_context,tmp])
        g += 1
    
    # year
    itemContext['year'] = list(map(extract_year,itemContext['releasedate']))
    itemContext['year_fact'], _ = pd.factorize(itemContext['year'])
    year = itemContext[['item','year_fact']]
    year.is_copy = False
    year.columns = ['item','user']
    year['user'] += numUser + g
    year['rating'] = data['rating'].mean()
    year['cv'] = -1
    data_context = pd.concat([data_context,age])
    
    current_perf = dict.fromkeys(['model','lambda','rank','crossval'])
    for elt in itertools.product(*[lambdas,rank,range(K)]):
        l, r, k = elt
        current_perf['model'] = 'contextual ALS-WR'
        current_perf['lambda'] = l
        current_perf['rank'] = r
        current_perf['crossval'] = k
        print(current_perf)
        bob = ALS(d=r,num_users=numUser,num_items=numItem,lbda=l)
        t0 = time()
        bob.fit(data_context[data_context['cv']!=k])
        T = time()-t0
        Rhat = bob.U.dot(bob.V.T)
        R_test = sparseMatrix(data_context,k,include=True)
        rmse = RMSE(R_test,Rhat)
        print(rmse)
        ind = getLine_fromdict(perf,current_perf)
        perf.loc[ind,['model','rank','lambda','crossval','rmse','runningTime']] = ['contextual ALS-WR',r,l,k,rmse,T]
    print('-'*50)


#==============================================================================
# improved (?) ALS-WR
#==============================================================================
if params['impALS-WR']['learn']=="True":
    print("Learn Improved ALS-WR")
    lambdas = params['impALS-WR']['params']['lambda']
    lambdas2 = params['impALS-WR']['params']['lambda2']
    rank = params['impALS-WR']['params']['rank']
    
    current_perf = dict.fromkeys(['model','lambda','lambda2','rank','crossval'])
    for elt in itertools.product(*[lambdas,lambdas2,rank,range(K)]):
        l, l2, r, k = elt
        current_perf['model'] = 'impALS-WR'
        current_perf['lambda'] = l
        current_perf['lambda2'] = l2
        current_perf['rank'] = r
        current_perf['crossval'] = k
        print(current_perf)
        bob = ALS(d=r,num_users=numUser,num_items=numItem,lbda=l,lbda2=l2,improved=True)
        t0 = time()
        bob.fit(data[data['cv']!=k])
        T = time()-t0
        Rhat = bob.U.dot(bob.V.T) 
        Rhat += np.repeat(bob.C,len(bob.D)).reshape(len(bob.C),len(bob.D))
        Rhat += np.repeat(bob.D,len(bob.C)).reshape(len(bob.D),len(bob.C)).T
        R_test = sparseMatrix(data,k,include=True)
        rmse = RMSE(R_test,Rhat)
        print(rmse)
        ind = getLine_fromdict(perf,current_perf)
        perf.loc[ind,['model','rank','lambda','lambda2','crossval','rmse','runningTime']] = ['impALS-WR',r,l,l2,k,rmse,T]
    print('-'*50)

#==============================================================================
# graphALS
#==============================================================================
if params['graphALS']['learn']=="True":
    print("Learn graphALS")
    lambdas = params['graphALS']['params']['lambda']
    rank = params['graphALS']['params']['rank']
    
    #==========================================================================
    # User Context Sparse Graph
    #==========================================================================
    # construct the laplacian using this coordinate :
    # scipy.sparse.csgraph.laplacian(scipy.sparse.csr_matrix(...))
    
    # occupation
    x = []
    y = []
    r = []
    for occ in set(userContext['occupation']):
        userIds = userContext['user'][userContext['occupation']==occ]
        x.extend(list(np.repeat(list(userIds),len(userIds))))
        y.extend(list(np.tile(list(userIds),len(userIds))))
        r.extend(list(np.repeat(1.,len(userIds)*len(userIds))))
    tmp = np.array([x,y,r]).T
    #tmp = pd.DataFrame([x,y,r]).T
    tmp = tmp[tmp[:,0]!=tmp[:,1]]
    userContextSparseGraph = tmp.T
    
    # age_slice
#==============================================================================
#     userContext['age_slice'] = pd.cut(userContext['age'],range(0,100,5))
#     x = []
#     y = []
#     r = []
#     for age in set(userContext['age_slice']):
#         userIds = userContext['user'][userContext['age_slice']==age]
#         x.extend(list(np.repeat(list(userIds),len(userIds))))
#         y.extend(list(np.tile(list(userIds),len(userIds))))
#         r.extend(list(np.repeat(1.,len(userIds)*len(userIds))))
#     tmp = np.array([x,y,r]).T
#     #tmp = pd.DataFrame([x,y,r]).T
#     tmp = tmp[tmp[:,0]!=tmp[:,1]]
#     userContextSparseGraph = np.concatenate((userContextSparseGraph,tmp.T),axis=1)
#==============================================================================
    
    #==========================================================================
    # Item Context Sparse Graph
    #==========================================================================
    
    # genre
    x = []
    y = []
    r = []
    
    for genre in genres:
        itemIds = itemContext['item'][itemContext[genre]==1]
        x.extend(list(np.repeat(list(itemIds),len(itemIds))))
        y.extend(list(np.tile(list(itemIds),len(itemIds))))
        r.extend(list(np.repeat(1.,len(itemIds)*len(itemIds))))
    tmp = np.array([x,y,r]).T
    #tmp = pd.DataFrame([x,y,r]).T
    tmp = tmp[tmp[:,0]!=tmp[:,1]]
    itemContextSparseGraph = tmp.T
    
    # year
#==============================================================================
#     itemContext['year'] = list(map(extract_year,itemContext['releasedate']))
#     x = []
#     y = []
#     r = []
#     
#     for year in set(itemContext['year']):
#         itemIds = itemContext['item'][itemContext['year']==year]
#         x.extend(list(np.repeat(list(itemIds),len(itemIds))))
#         y.extend(list(np.tile(list(itemIds),len(itemIds))))
#         r.extend(list(np.repeat(1.,len(itemIds)*len(itemIds))))
#     tmp = np.array([x,y,r]).T
#     #tmp = pd.DataFrame([x,y,r]).T
#     tmp = tmp[tmp[:,0]!=tmp[:,1]]
#     itemContextSparseGraph = np.concatenate((itemContextSparseGraph,tmp.T),axis=1)
#==============================================================================
    
    current_perf = dict.fromkeys(['model','lambda','rank','crossval'])
    for elt in itertools.product(*[lambdas,rank,range(K)]):
        l, r, k = elt
        current_perf['model'] = 'graphALS'
        current_perf['lambda'] = l
        current_perf['rank'] = r
        current_perf['crossval'] = k
        print(current_perf)
        bob = graphALS(d=r,num_users=numUser,num_items=numItem,lbda=l,
                        solve='jacobi',verbose=True)
        t0 = time()
        bob.fit(train = data[data['cv']!=k],
                userGraph = userContextSparseGraph,
                itemGraph = itemContextSparseGraph)
        T = time()-t0
        Rhat = bob.U.dot(bob.V.T)
        R_test = sparseMatrix(data,k,include=True)
        W = R_test.nonzero()
        rmse = RMSE(R_test,Rhat)
        print(rmse)
        ind = getLine_fromdict(perf,current_perf)
        perf.loc[ind,['model','rank','lambda','crossval','rmse','runningTime']] = ['graphALS',r,l,k,rmse,T]
    print('-'*50)

#==============================================================================
# multiple graphALS
#==============================================================================
if params['multipleGraphALS']['learn']=="True":
    print("Learn multipleGraphALS")
    lambdas = params['graphALS']['params']['lambda']
    rank = params['graphALS']['params']['rank']
    
    #==========================================================================
    # User Context Sparse Graph
    #==========================================================================
    # construct the laplacian using this coordinate :
    # scipy.sparse.csgraph.laplacian(scipy.sparse.csr_matrix(...))
    
    userContextSparseGraph = {}
    # occupation
    x = []
    y = []
    r = []
    for occ in set(userContext['occupation']):
        userIds = userContext['user'][userContext['occupation']==occ]
        x.extend(list(np.repeat(list(userIds),len(userIds))))
        y.extend(list(np.tile(list(userIds),len(userIds))))
        r.extend(list(np.repeat(1.,len(userIds)*len(userIds))))
    tmp = np.array([x,y,r]).T
    #tmp = pd.DataFrame([x,y,r]).T
    tmp = tmp[tmp[:,0]!=tmp[:,1]]
    
    userContextSparseGraph['occupation'] = tmp.T
    
    #==========================================================================
    # Item Context Sparse Graph
    #==========================================================================
    
    itemContextSparseGraph = {}
    # item
    x = []
    y = []
    r = []
    
    for genre in genres:
        itemIds = itemContext['item'][itemContext[genre]==1]
        x.extend(list(np.repeat(list(itemIds),len(itemIds))))
        y.extend(list(np.tile(list(itemIds),len(itemIds))))
        r.extend(list(np.repeat(1.,len(itemIds)*len(itemIds))))
    tmp = np.array([x,y,r]).T
    #tmp = pd.DataFrame([x,y,r]).T
    tmp = tmp[tmp[:,0]!=tmp[:,1]]
    
    itemContextSparseGraph['genre'] = tmp.T
    keys = ['occupation','genre']
    
    current_perf = dict.fromkeys(['model','lambda','rank','crossval'])
    for elt in itertools.product(*[lambdas,rank,range(K)]):
        l, r, k = elt
        current_perf['model'] = 'blend-multipleGraphALS'
        current_perf['lambda'] = l
        current_perf['rank'] = r
        current_perf['crossval'] = k
        print(current_perf)
        bob = multipleGraphALS(d=r,num_users=numUser,num_items=numItem,lbda=l,verbose=True)
        t0 = time()
        bob.fit(train = data[data['cv']!=k],
                userGraph = userContextSparseGraph,
                itemGraph = itemContextSparseGraph)
        T = time()-t0
        Rhat = {}
        R_test = sparseMatrix(data,k,include=True)
        W = R_test.nonzero()
        blender = pd.DataFrame({'user':W[0],'item':W[1]})
        for key in keys:
            Rhat[key] = bob.graph[key].U.dot(bob.graph[key].V.T)
            rmse = RMSE(R_test,Rhat[key],W)
            print(key,'-',rmse)

            blender[key] = Rhat[key][W]
        R = sparseMatrix(data,k,include=False)
        n_u = pd.DataFrame({'user':range(numUser),'n_u':list(map(lambda u: R[u,:].nnz,range(numUser)))})
        n_i = pd.DataFrame({'item':range(numItem),'n_i':list(map(lambda i: R[:,i].nnz,range(numItem)))})
        blender = pd.merge(pd.merge(blender,n_u),n_i)
        blender['r'] = np.array(R_test[W])[0]
        blender['cv'] = np.random.randint(0,K,blender.shape[0])
        blender['rhat'] = -1
        var = keys+['n_u','n_i']
        for kb in range(K):
            blend = RFC(n_estimators = 300)
            blend.fit(blender[var][blender['cv']!=kb],blender['r'][blender['cv']!=kb])
            blender.loc[np.where(blender['cv']==kb)[0],'rhat'] = blend.predict_proba(blender[var][blender['cv']==kb]).dot(np.array([1,2,3,4,5]))
        rmse = mean_squared_error(blender['r'],blender['rhat'])**0.5
        print(rmse)
        ind = getLine_fromdict(perf,current_perf)
        perf.loc[ind,['model','rank','lambda','crossval','rmse','runningTime']] = ['blend-multipleGraphALS',r,l,k,rmse,T]
    print('-'*50)

#==============================================================================
# RSVD
#==============================================================================
if params['RSVD']['learn']=="True":
    print("Learn RSVD")
    lambdas = params['RSVD']['params']['lambda']
    rank = params['RSVD']['params']['rank']
    
    current_perf = dict.fromkeys(['model','lambda','rank','crossval'])
    for elt in itertools.product(*[lambdas,rank,range(K)]):
        l, r, k = elt
        current_perf['model'] = 'RSVD'
        current_perf['lambda'] = l
        current_perf['rank'] = r
        current_perf['crossval'] = k
        print(current_perf)
        bob = RSVD(d=r,num_users=numUser,num_items=numItem,lbda=l)
        t0 = time()
        bob.fit(data[data['cv']!=k],batch_size = 100)
        T = time()-t0
        Rhat = bob.U.dot(bob.V.T)
        R_test = sparseMatrix(data,k,include=True)
        rmse = RMSE(R_test,Rhat)
        print(rmse)
        ind = getLine_fromdict(perf,current_perf)
        perf.loc[ind,['model','rank','lambda','crossval','rmse','runningTime']] = ['RSVD',r,l,k,rmse,T]
    print('-'*50)

#==============================================================================
# improved RSVD
#==============================================================================
if params['impRSVD']['learn']=="True":
    print("Learn Improved RSVD")
    lambdas = params['impRSVD']['params']['lambda']
    lambdas2 = params['impRSVD']['params']['lambda2']
    rank = params['impRSVD']['params']['rank']
    
    current_perf = dict.fromkeys(['model','lambda','lambda2','rank','crossval'])
    for elt in itertools.product(*[lambdas,lambdas2,rank,range(K)]):
        l, l2, r, k = elt
        current_perf['model'] = 'impRSVD'
        current_perf['lambda'] = l
        current_perf['lambda2'] = l2
        current_perf['rank'] = r
        current_perf['crossval'] = k
        print(current_perf)
        bob = RSVD(d=r,num_users=numUser,num_items=numItem,lbda=l,lbda2=l2,improved=True)
        t0 = time()
        bob.fit(data[data['cv']!=k])
        T = time()-t0
        Rhat = bob.U.dot(bob.V.T)
        Rhat += np.repeat(bob.C,len(bob.D)).reshape(len(bob.C),len(bob.D))
        Rhat += np.repeat(bob.D,len(bob.C)).reshape(len(bob.D),len(bob.C)).T
        R_test = sparseMatrix(data,k,include=True)
        rmse = RMSE(R_test,Rhat)
        print(rmse)
        ind = getLine_fromdict(perf,current_perf)
        perf.loc[ind,['model','rank','lambda','crossval','rmse','runningTime']] = ['impRSVD',r,l,k,rmse,T]
    print('-'*50)

#==============================================================================
# NNgraphMF
#==============================================================================
if params['NNgraphMF']['learn']=="True":
    print("Learn non negative graph MF")
    lambdas = params['NNgraphMF']['params']['lambda']
    rank = params['NNgraphMF']['params']['rank']
    
    #==========================================================================
    # User Context Sparse Graph
    #==========================================================================
    # construct the laplacian using this coordinate :
    # scipy.sparse.csgraph.laplacian(scipy.sparse.csr_matrix(...))
    
    # occupation
    x = []
    y = []
    r = []
    for occ in set(userContext['occupation']):
        userIds = userContext['user'][userContext['occupation']==occ]
        x.extend(list(np.repeat(list(userIds),len(userIds))))
        y.extend(list(np.tile(list(userIds),len(userIds))))
        r.extend(list(np.repeat(1.,len(userIds)*len(userIds))))
    userContextSparseGraph = pd.DataFrame([x,y,r]).T
    userContextSparseGraph = userContextSparseGraph[userContextSparseGraph[0]!=userContextSparseGraph[1]]
    
    #==========================================================================
    # Item Context Sparse Graph
    #==========================================================================
    
    # item
    x = []
    y = []
    r = []
    
    for genre in genres:
        itemIds = itemContext['item'][itemContext[genre]==1]
        x.extend(list(np.repeat(list(itemIds),len(itemIds))))
        y.extend(list(np.tile(list(itemIds),len(itemIds))))
        r.extend(list(np.repeat(1.,len(itemIds)*len(itemIds))))
    itemContextSparseGraph = pd.DataFrame([x,y,r]).T
    itemContextSparseGraph = itemContextSparseGraph[itemContextSparseGraph[0]!=itemContextSparseGraph[1]]
    
    current_perf = dict.fromkeys(['model','lambda','lambda2','rank','crossval'])
    for elt in itertools.product(*[lambdas,rank,range(K)]):
        l, r, k = elt
        current_perf['model'] = 'NNgraphMF'
        current_perf['lambda'] = l
        current_perf['rank'] = r
        current_perf['crossval'] = k
        print(current_perf)
        bob = NNGraphMF(d=r,num_users=numUser,num_items=numItem,lbda=l,verbose=True)
        t0 = time()
        bob.fit(data[data['cv']!=k],userContextSparseGraph,itemContextSparseGraph)
        T = time()-t0
        Rhat = bob.U.dot(bob.V.T)
        R_test = sparseMatrix(data,k,include=True)
        rmse = RMSE(R_test,Rhat)
        print(rmse)
        ind = getLine_fromdict(perf,current_perf)
        perf.loc[ind,['model','rank','lambda','lambda2','crossval','rmse','runningTime']] = ['NNgraphMF',r,l,l2,k,rmse,T]
    print('-'*50)