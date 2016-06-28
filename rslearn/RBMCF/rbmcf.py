""" 
Code in development, still not sure of what comes out :)
For now the code is not very flexible since it require as input a pandas Data Frame
with the right columns name : u, i, r for respectively user, item and rating

to do : 
- modify learning with momemtum and weight decay
- add comments
"""

import tensorflow as tf
import math
import numpy as np
import pandas as pd

# for mini batch construction
def chunks(l, nbatch):
    n = int(np.ceil(len(l)/nbatch))
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i+n]

# construct the placeholder
def placeholder_inputs():
    user_placeholder = tf.placeholder(tf.int64,[None,2])
    M_placeholder = tf.placeholder(tf.int64,[1])
    return user_placeholder, M_placeholder

# deal with the feed_dict
def fill_feed_dict(R, u, user_pl, M_pl):
    D = R[R['u']==u][['i','r']]
    D = D.sort_values(['i','r'])
    D = np.array(D).tolist()
    M = [len(D)]
    feed_dict = {
        user_pl: D,
        M_pl: M
    }

    return feed_dict

# convert the DataFrame to a sparse matrix for each user
def createV(visible_arr, M, m, K):
    """ 
    visible_arr : array with the item x rating coordinate 
    (i.e movie 1, rating 4 and movie 2 rating 3 : [[1,4],[2,3]])
    M : number of movies rated by active user
    m : total number of movies (setting in FLAGS ?)
    K : number of possibles ratings (setting in FLAGS ?)
    """
    M = tf.cast(M,tf.int32)
    V = tf.SparseTensor(indices = visible_arr, values = tf.tile([1.0],M),
                        shape = [m,K])
    return V

def sample_prob(probs):
    """Takes a tensor of probabilities (as from a sigmoidal activation)
       and samples from all the distributions
    See A Practical Guide to Training Restricted Boltzmann Machines part 3 of Hinton 
    for more details """
    return tf.nn.relu(
        tf.sign(
            probs - tf.random_uniform(probs.get_shape())))
            
def is_known(V):
    """ check is the movie has been rated by the user, 
    so check if the column i of have one positive value or not
    """
    return tf.reduce_sum(tf.sparse_tensor_to_dense(V),1)

# if not sparse, tranpose, else, dense it then transpose
def transpose2(T):
    try:
        return tf.transpose(T)
    except TypeError:
        return tf.transpose(tf.sparse_tensor_to_dense(T))

def cummatmulK(V,W,m,K,H):
    """ compute the vector h such that h[j] = \sum_i \sum_k v_i^k W_{i,j}^k """
    return tf.matmul(tf.reshape(V,shape=[1,m*K]),
                      tf.reshape(W,shape=[m*K,H]))

def matmul3D(H,W,K):
    for k in range(K):
        if k==0:
            A = tf.matmul(H,W[k,:,:],transpose_b=True)
        else:
            A = tf.concat(0,[A,tf.matmul(H,W[k,:,:],transpose_b=True)])
    return tf.transpose(A)

def matmul_to3D(A,B,m,F,K):
    for k in range(K):
        if k==0:
            col = tf.reshape(B[:,0],(m,1))
            out = tf.matmul(col,A)
        else:
            col = tf.reshape(B[:,k],(m,1))
            out = tf.concat(0,[out,tf.matmul(col,A)])
    return tf.reshape(out,(K,m,F))

def softmax_filtered(num,ind,m,K):
    """ num : (m,K) matrix corresponding to b_i^k + \sum_j h_j W_{ij}^k
    ind : index of the rated motives
    m : total number of movies
    K : number of possible ratings (i.e if from 1 to 5, K = 5) """
    ind = tf.transpose(tf.reshape(tf.tile(ind,[K]),(K,m)))
    num = tf.mul(tf.exp(num),ind) # filter on only rated movies
    den = tf.reduce_sum(num,0)
    den = tf.reshape(tf.tile(den,[m]),(m,K))
    return tf.div(num,den)

class RBM(object):
    """ represents a sigmoidal rbm """

    def __init__(self, name, num_users, m, F, K):
        """ using the notation of RBM for CF 
        m : number of movies
        F : number of hidden units
        K : number of values that the rating can take
        """
        with tf.name_scope("rbm_" + name):
            self.sess = tf.Session()
            self.num_users = num_users
            self.m = m
            self.F = F
            self.K = K
            self.weights = tf.Variable(
                tf.truncated_normal([K,m,F],
                    stddev=1.0 / math.sqrt(float(m))), name="weights")
            self.v_bias = tf.Variable(tf.zeros([m,K]), name="v_bias")
            self.h_bias = tf.Variable(tf.zeros([1,F]), name="h_bias")
            self.delta_w = tf.Variable(tf.zeros([self.K,self.m,self.F]))
            self.delta_vb = tf.Variable(tf.zeros([self.m,self.K]))
            self.delta_hb = tf.Variable(tf.zeros([1,self.F]))

    def propup(self, visible):
        """ P(h|v) - formula (2) of the paper
        visible is a m x K sparse tensor 
        the unknown movie has a row full of zero so won't impact the computation
        compute the sum of V_k*W_k for each slice k
        """
        return tf.nn.sigmoid(cummatmulK(transpose2(visible), self.weights, self.m, self.K, self.F) + self.h_bias)

    def propdown(self, hidden, visible):
        """ P(v|h) - formula (1) of the paper
        return a value for a each slice k of H*W_k"""
        return softmax_filtered(matmul3D(hidden, self.weights, self.K) + self.v_bias, 
                        is_known(visible),self.m, self.K)

    def sample_h_given_v(self, v_sample):
        """ Generate a sample from the hidden layer """
        return sample_prob(self.propup(v_sample))

    def sample_v_given_h(self, h_sample, visible):
        """ Generate a sample from the visible layer """
        return sample_prob(self.propdown(h_sample, visible))

    def gibbs_hvh(self, h0_sample, visible):
        """ A gibbs step starting from the hidden layer """
        v_sample = self.sample_v_given_h(h0_sample, visible)
        h_sample = self.sample_h_given_v(v_sample)
        return [v_sample, h_sample]

    def gibbs_vhv(self, v0_sample):
        """ A gibbs step starting from the visible layer """
        h_sample = self.sample_h_given_v(v0_sample)
        v_sample = self.sample_v_given_h(h_sample)
        return  [h_sample, v_sample]

#==============================================================================
#     def cd1(self, visibles, learning_rate=0.01):
#         h_start = self.propup(visibles)
#         v_end = self.propdown(h_start,visibles)
#         h_end = self.propup(v_end)
#         # adapt to 3D tensor here
#         w_positive_grad = matmul_to3D(h_start, tf.sparse_tensor_to_dense(visibles), self.m, self.F, self.K) # formula (5) of the paper
#         w_negative_grad = matmul_to3D(h_end, v_end, self.m, self.F, self.K) # formula (5) of the paper
#
#         update_w = self.delta_w.assign_add(learning_rate/self.num_users * (w_positive_grad - w_negative_grad))
# 
#         update_vb = self.delta_vb.assign_add(learning_rate/self.num_users * (tf.sparse_tensor_to_dense(visibles) - v_end))
# 
#         update_hb = self.delta_hb.assign_add(learning_rate/self.num_users * (h_start - h_end))
# 
#         return [update_w, update_vb, update_hb]
#==============================================================================
    
    def cdk(self, visibles, k, learning_rate=0.001):
        h_start = self.propup(visibles)
        h_t = h_start
        
        for t in range(k):
            v_t = self.propdown(h_t,visibles)
            h_t = self.propup(v_t)
        # adapt to 3D tensor here
        w_positive_grad = matmul_to3D(h_start, tf.sparse_tensor_to_dense(visibles), self.m, self.F, self.K) # formula (5) of the paper
        w_negative_grad = matmul_to3D(h_t, v_t, self.m, self.F, self.K) # formula (5) of the paper
        
        update_w = self.delta_w.assign_add(learning_rate * (w_positive_grad - w_negative_grad))

        update_vb = self.delta_vb.assign_add(learning_rate * (tf.sparse_tensor_to_dense(visibles) - v_t))

        update_hb = self.delta_hb.assign_add(learning_rate * (h_start - h_t))

        return [update_w, update_vb, update_hb]
    
    
    def update_weight(self):
        update_w = self.weights.assign_add(self.delta_w)
        update_vb = self.v_bias.assign_add(self.delta_vb)
        update_hb = self.h_bias.assign_add(self.delta_hb)
        
        return [update_w, update_vb, update_hb]
        
    def train(self,v_arr,M,k,learning_rate=0.001):
        """ perform update for one user """
        V = createV(v_arr,M,self.m, self.K)
        # then perform cd1 
        a, b, c = self.cdk(V,k,learning_rate)
        return a, b, c
        
    def fit(self, dataset, n_epochs = {1:10,3:10,5:10}, n_mini_batch = 1, learning_rate=0.01):
        """ learn parameters by performing n_epochs loop """
        self.data = dataset
        user_placeholder, M_placeholder = placeholder_inputs()
        self.sess.run(tf.initialize_all_variables()) 
        
        tot_epochs = 0
        for k in n_epochs.keys():
            print("perform CD",k)
            
            for epochs in range(n_epochs[k]):
                tot_epochs+=1
                print("epochs",tot_epochs)
                user_batchs = chunks(range(self.num_users),n_mini_batch)
                
                b = 0
                for batch in user_batchs:
                    b+=1
                    print("batch :",b,"/",n_mini_batch)
                    # the learning rate is divided by the batch-size
                    # the last batch does not necesarilly have the same size as 
                    # the offer, so we have to init train_op here
                    train_op = self.train(user_placeholder,M_placeholder,k,learning_rate/len(batch))
                    update_op = self.update_weight()
                    
                    # re-initialize the gradient
                    self.sess.run(tf.initialize_variables([self.delta_w,self.delta_vb,self.delta_hb]))
                    for u in batch:
                        feed_dict = fill_feed_dict(self.data, u, user_placeholder, M_placeholder)
                        # update the gradient
                        self.sess.run(train_op, feed_dict = feed_dict)
                    # update the weight for this mini-batch
                    self.sess.run(update_op)
    
#==============================================================================
#     def get_prediction(self,item,v_arr,M):
#         V = createV(v_arr,M,self.m, self.K)
#         V = tf.sparse_tensor_to_dense(V)
#         R = 0 # hold the probability the user give a rate of k to the item
#         den = 0
#         # There are two terms in the exponential that are common to each k
#         Exp_c = self.h_bias
#         for l in range(self.K):
#             Exp_c = tf.add(Exp_c,tf.matmul(tf.reshape(V[:,l],(1,self.m)),self.weights[l,:,:]))
#         
#         Gamma = tf.mul(V[item,:],self.v_bias[item,:]) 
#         
#         for k in range(self.K):
#             tmp = tf.add(Exp_c,V[item,k]*self.weights[k,item,:])
#             tmp = tf.add(tf.exp(tmp),tf.ones([self.F],dtype=tmp.dtype))
#             tmp = tf.reduce_prod(tmp)
#             tmp = Gamma[k]*tmp
#             den = den + tmp
#             R = R + (k+1)*tmp
#         R = R/den
#         
#         return R
#     
#     def predict(self,user,item):
#         user_placeholder, M_placeholder = placeholder_inputs()
#         p = self.get_prediction(item, user_placeholder,M_placeholder)
#         feed_dict = fill_feed_dict(self.data, user, user_placeholder, M_placeholder)
#         res = self.sess.run(p, feed_dict = feed_dict)
#         return res
#==============================================================================
    
    def get_prediction(self,items,v_arr,M):
        """ items contains either a list of item or a single item
        for instance items = [0,1,2] or items = 0
        when predicting multiple value for the same user, it is more efficient 
        to use this function with a list of some values in items instead of 
        calling this function multiple times since it prevent from computing 
        the same V for each rating prediction        
        """
        V = createV(v_arr,M,self.m, self.K)
        V = tf.sparse_tensor_to_dense(V)
        
        den = 0
        # There are two terms in the exponential that are common to each k
        Exp_c = self.h_bias
        for l in range(self.K):
            Exp_c = tf.add(Exp_c,tf.matmul(tf.reshape(V[:,l],(1,self.m)),self.weights[l,:,:]))
        
        if (type(items)==list) | (type(items)==np.ndarray) | (type(items)==range):
            R = [] 
            if (type(items)!=range):
                items = list(map(int,items)) # it must be int and not np.int32
                
            for item in items:        
            
                Gamma = tf.exp(tf.mul(V[item,:],self.v_bias[item,:]))
                
                R_tmp = 0
                for k in range(self.K):
                    tmp = tf.add(tf.exp(-Exp_c),tf.exp(V[item,k]*self.weights[k,item,:]))
                    tmp = tf.reduce_prod(tmp)
                    tmp = tf.reduce_prod(tmp)
                    tmp = Gamma[k]*tmp
                    den = den + tmp
                    R_tmp = R_tmp + (k+1)*tmp
                R_tmp = R_tmp/den
                R.extend([R_tmp])
                
        elif type(items)==int:
            R = 0
            Gamma = tf.exp(tf.mul(V[items,:],self.v_bias[items,:]))

            for k in range(self.K):
                tmp = tf.add(tf.exp(-Exp_c),tf.exp(V[items,k]*self.weights[k,items,:]))
                tmp = tf.reduce_prod(tmp)
                tmp = tf.reduce_prod(tmp)
                tmp = Gamma[k]*tmp
                den = den + tmp
                R = R + (k+1)*tmp
            R = R/den
        else:
            print('type error')
            
        return R
    
    def predict(self,user,items):
        """ items contains either a list of item or a single item
        for instance items = [0,1,2] or items = 0
        when predicting multiple value for the same user, it is more efficient 
        to use this function with a list of some values in items instead of 
        calling this function multiple times since it prevent from computing 
        the same V for each rating prediction        
        """
        user_placeholder, M_placeholder = placeholder_inputs()
        ps = self.get_prediction(items, user_placeholder,M_placeholder)
        feed_dict = fill_feed_dict(self.data, user, user_placeholder, M_placeholder)
        res = self.sess.run(ps, feed_dict = feed_dict)
        return res    
    
    def predict_all(self,df):
        """ df must contains a data frame with two columns : 
        - u : the user 
        - i : the item 
        the function will call the predict function for each user with items 
        containing all the desired item for that user. So you don't need to 
        care about the order to be efficient
        This will return a data frame with an extra column 'r', containing the prediction
        """
        
        out = pd.DataFrame(columns=('u','i','r_hat'))        
        users = list(set(df['u']))
        for user in users:
            items = list(df['i'][df['u']==user])
            r = self.predict(user,items)
            tmp = pd.DataFrame({'u':np.repeat(user,len(items)),
                                'i':items,
                                'r_hat':r})
            out = out.append(tmp)
        return out
        
        
    def reconstruction_error(self, dataset):
        """ The reconstruction cost for the whole dataset """
        err = tf.stop_gradient(dataset - self.gibbs_vhv(dataset)[1])
        return tf.reduce_sum(err * err)

