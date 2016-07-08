import numpy as np
import scipy.sparse as sparse
from copy import deepcopy

def Jacobi(A, b, max_iter = 1000):
    n = A.shape[0]
    D1 = 1/np.diag(A)

    R = deepcopy(A)
    R[range(n),range(n)] = 0
    x = np.zeros_like(b)
    x_new = np.zeros_like(b)
    
    for it_count in range(max_iter):
        x_new =np.diag(D1).dot(b-R.dot(x))
    
        if np.allclose(x, x_new, atol=1e-5):
            break
   
        x = x_new
    return x

def spJacobi(A, b, max_iter = 1000):

    n = A.shape[0]
    D1 = 1/A.diagonal()
    D1 = sparse.dia_matrix((D1,[0]),shape=(n,n))
    
    R = deepcopy(A)
    R.setdiag(0)
    x = np.zeros(b.shape)
    x_new = np.zeros(b.shape)
    
    for it_count in range(max_iter):
        x_new = D1.dot(b-R.dot(x))
         
        if np.allclose(x, x_new, atol=1e-5):
            print(it_count)
            break
   
        x = x_new
    return x
