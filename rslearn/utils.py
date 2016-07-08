from sklearn.utils import as_float_array
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

def sparse_matrix_input(X):
    X = as_float_array(X)[:,:3]
    return X

def sparse_matrix(X,n,p,w=1):
    #X = sparse_matrix_input(X)
    R = csr_matrix((w*X['val'], (X['row'],X['col'])), shape=(n,p))
    return R

def RMSE(R_true,R_pred,W=None):
    if W is None:
        W = R_true.nonzero()
    if 'sparse' in str(type(R_true)):
        return mean_squared_error(np.array(R_true[W])[0],R_pred[W])**.5
    else:
        return mean_squared_error(R_true[W],R_pred[W])**.5

def rankMeasure(R_true,R_pred):
    numUser, numItem = R_true.shape
    num = 0
    den = 0
    
    y = list(R_true.indices)
    indptr = R_true.indptr
    data = list(R_true.data)
    n_u = indptr[1:len(indptr)]-indptr[0:(len(indptr)-1)]

    for u in range(numUser):
        if n_u[u]>0:
            items = y[indptr[u]:indptr[u+1]]
            
            rt_ui = data[indptr[u]:indptr[u+1]]
            rhat = R_pred[u,items]
            rank_ui = list(-np.sort(-rhat))
            r = map(lambda x: rank_ui.index(x)/n_u[u],rhat)
            r =[a*b for a,b in zip(r,rt_ui)]
            r = 100*np.round(sum(r),2)
            num+=r
            den+=np.sum(rt_ui)

    return num/den

def sparseMatrix(data,k,include=False,names=['user','item','rating']):
    # if include = True we take only cv=k, ortherwise we only exclude cv=k
    n = len(set(data[names[0]]))
    p = len(set(data[names[1]]))
    if include:
        R = csr_matrix((data[names[2]][data['cv']==k], 
                                (data[names[0]][data['cv']==k], 
                                 data[names[1]][data['cv']==k])),
                         shape=(n,p))
    else:
        R = csr_matrix((data[names[2]][data['cv']!=k], 
                                (data[names[0]][data['cv']!=k], 
                                 data[names[1]][data['cv']!=k])),
                         shape=(n,p))
    return R

def getLine(perf,model,param,cv):
    try: # if the line exists
        out = perf[(perf['model']==model) & (perf['params']==param) & 
            (perf['crossval']==cv)].index[0]
    except IndexError: # create a new line
        try: # the dataset is not empty
            out = max(perf.index)+1
        except ValueError:
            out = 0
    return out
    
def getLine_fromdict(perf,Dict):
    tmp = pd.DataFrame(columns=list(Dict.keys()))
    tmp.loc[0,list(Dict.keys())] = list(Dict.values())
    tmp = pd.merge(perf,tmp)
    try: # the dataset is not empty
        out = max(perf.index)+1
    except ValueError:
        out = 0
    return out

def extract_year(x):
    try:
        return int(x.split('-')[2])
    except AttributeError:
        return -1

def argmax(x):
    p = np.argwhere(x==np.max(x))
    return p[np.random.randint(len(p))][0]