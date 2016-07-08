import pandas as pd
import numpy as np
from random import shuffle, seed
from rslearn.graphALS.graphALS import graphALS
from rslearn.ALSWR.alswr import ALS
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_squared_error

# fix all the random seeds
seed(0)
np.random.seed(0)

# loading data
data = pd.read_csv("data/ml100k.data",
                   sep="\t",index_col=False,
                   names=['user','movie','rating','timestamp'])

userContext = pd.read_csv("data/u.user",sep="|",index_col=False,
                       names=['user','age','gender','occupation','zipcode'])
    
itemContext = pd.read_csv("data/u.item",sep="|",index_col=False, encoding = "ISO-8859-1",
                   names=['movie','movietitle','releasedate',
                   'videoreleasedate','IMDbURL','unknown','Action',
                   'Adventure','Animation','Children','Comedy','Crime',
                   'Documentary','Drama','Fantasy','FilmNoir','Horror',
                   'Musical','Mystery','Romance','SciFi','Thriller','War',
                   'Western'])

genres = ['Action','Adventure','Animation','Children','Comedy','Crime',
          'Documentary','Drama','Fantasy','FilmNoir','Horror','Musical',
          'Mystery','Romance','SciFi','Thriller','War','Western']

#==============================================================================
# filter out some user and items
#==============================================================================

# subset user
countsUser = data['user'].value_counts()
selectedUser = pd.DataFrame({'user':list(countsUser.index[countsUser>50]),
                             'count':list(countsUser[countsUser>50])})
data = pd.merge(data,selectedUser)

# subset item
countsItem = data['movie'].value_counts()
selectedItem = pd.DataFrame({'movie':list(countsItem.index[countsItem>50])})
data = pd.merge(data,selectedItem)

userTab = pd.DataFrame({'user': np.unique(data['user']),
                       'userId': np.arange(len(np.unique(data['user'])))})
                       
itemTab = pd.DataFrame({'movie': np.sort(np.unique(data['movie'])),
                       'movieId': np.arange(len(np.unique(data['movie'])))})

data = pd.merge(pd.merge(data,userTab),itemTab)
data = data[['userId','movieId','rating']]

#==============================================================================
# Subset the context dataframe
#==============================================================================
userContext = pd.merge(userTab,userContext)
itemContext = pd.merge(itemTab,itemContext)

numUser = userTab.shape[0]
numItem = itemTab.shape[0]

#==============================================================================
# User Context Sparse Graph
#==============================================================================
# construct the laplacian using this coordinate :
# scipy.sparse.csgraph.laplacian(scipy.sparse.csr_matrix(...))

# occupation
x = []
y = []
r = []
for i in np.arange(userContext.shape[0]):
    occ = userContext['occupation'][i]
    associated = userContext['userId'][(userContext['occupation']==occ) & (userContext['userId']!=i)]
    x.extend(list(np.repeat(i,len(associated))))
    y.extend(list(associated))
    r.extend(list(np.repeat(1.,len(associated))))

userContextSparseGraph = pd.DataFrame([x,y,r]).T

#==============================================================================
# Item Context Sparse Graph
#==============================================================================

# item
x = []
y = []
r = []
for i in np.arange(itemContext.shape[0]):
    genres_i = itemContext[genres].loc[i]
    genres_i = list(genres_i.index[genres_i==1])
    for genre in genres_i:
        associated = itemContext['movieId'][(itemContext[genre]==1) & (itemContext['movieId']!=i)]
        x.extend(list(np.repeat(i,len(associated))))
        y.extend(list(associated))
        r.extend(list(np.repeat(1.,len(associated))))

itemContextSparseGraph = pd.DataFrame([x,y,r]).T

#==============================================================================
# Error measure
#==============================================================================
def RMSE(R_true,R_pred,W):
    return mean_squared_error(np.array(R_true[W])[0],R_pred[W])**.5

#==============================================================================
# Train/Test split
#==============================================================================
shuffleUserId = np.arange(numUser)
shuffle(shuffleUserId)
shuffleUserId = list(shuffleUserId)

# K cross validation
K = 5
KshuffleUserId = []
start = 0
end = int(numUser/K)
L = int(numUser/K)
for k in range(K):
    if k!=(K-1):
        KshuffleUserId.append(shuffleUserId[start:end])
        start = end
        end+=L
    else:
        KshuffleUserId.append(shuffleUserId[start:])

for k in range(K):
    print("cross-val",k)
    trainUser = []
    testUser = []
    for i in range(K):
        if i==k:
            testUser = KshuffleUserId[i]
        else:
            trainUser.extend(KshuffleUserId[i])
    
    data['rnd'] = np.random.random(data.shape[0])
    
    trainInd = (list(map(lambda x: x in trainUser,data['userId'])) |
                (data['rnd']<0.7))

    dataTrain = data[trainInd]
    dataTest = data[np.invert(trainInd)]
    
    bob = graphALS(num_factors = 5,num_iterations = 20,verbose = True)
    bob.fit(dataTrain[['userId','movieId','rating']],
                   userContextSparseGraph,itemContextSparseGraph)
    Rhat = bob.user_vectors.dot(bob.item_vectors.T)
    R_test = csr_matrix((dataTest['rating'],(dataTest['userId'], dataTest['movieId'])),
                         shape=(numUser,numItem))
    W = R_test.nonzero()
    rmse = RMSE(R_test,Rhat,W)
    print(rmse)
    R = csr_matrix((dataTrain['rating'],(dataTrain['userId'], dataTrain['movieId'])),
                         shape=(numUser,numItem))
    bobALS = ALS(5)
    bobALS.fit(R)
    Rhat_ALS = bobALS.U.dot(bobALS.V.T)
    rmseALS = RMSE(R_test,Rhat_ALS,W)
    print(rmseALS)