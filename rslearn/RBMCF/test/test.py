from recsyslib.RBMCF.rbmcf import RBM
import pandas as pd
import numpy as np
from random import sample

data = pd.read_csv("data/ml100k.data",sep="\t",index_col=False,
                       names=['u','i','r','t'])

data['u'] -= 1 # to start at 0
data['i'] -= 1
data['r'] -= 1
num_users = len(set(data['u']))
num_items= len(set(data['i']))

data['split'] = np.random.rand(data.shape[0])
dataTrain = data[['u','i','r']][data['split']>0.1]
if len(set(dataTrain['u']))!=num_users:
    print("some users don't appear in the training set ... ")

dataTest = data[['u','i','r']][data['split']<=0.1]

# should increase F and n_epochs
bob = RBM(name='ML100k', num_users = num_users, m = num_items, F = 100, K = 5)
bob.fit(dataTrain, n_epochs = {1:10,3:10,5:10}, n_mini_batch = 2)

# predict a particular item
bob.predict(user=0,items=0)
bob.predict(user=0,items=[0]) # will return a list 

# predict multiple items for the same user
bob.predict(user=0,items=[0,1,2])
bob.predict(user=0,items=range(5))
bob.predict(user=0,items=np.arange(5))

# predict for a whole set of users and items
ind = sample(range(dataTest.shape[0]),100)
A = dataTest.iloc[ind][['u','i']]
Rhat = bob.predict_all(A)
Rhat['r_hat']-=1 # because we shifted it before for the test set

Rhat = pd.merge(Rhat,dataTest,on=['u','i'])
rmse = np.sqrt(np.mean((Rhat['r_hat']-Rhat['r'])**2))
print(rmse)