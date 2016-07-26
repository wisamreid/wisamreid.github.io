"""
Original by Andrej Kaparthy, Fei Fei Li, and Justin Johnson
modified by Iran Roman
"""

import numpy as np

class KNN(object):
    """ a kNN classifier with L2 and L1 distances """

    def __init__(self):
        pass

    def train(self, X, y):

        # TODO: train the algorithm
        
        self.X_train = X
        self.y_train = y
    
    def predict(self, X, k=1, L=1):

        dists = self.compute_distances(X, L=1)

        return self.predict_labels(dists, k=k)

    def compute_distances(self, X, L=1):
        
    ########
    ## Fully vectorized implementation
    ########

        num_training = self.X_train.shape[0]
        num_test = X.shape[0]

        dists = np.zeros((num_test, num_training))

        X_2nd_pow=X**2
        X_train_2nd_pow=self.X_train ** 2
    
        # sum over the rows (each data vector) of this matrices to obtain the
        # magnitude squared of each vector.
        X_pts_mag_sqrd = np.sum(X_2nd_pow, axis=1,keepdims=True)
        X_train_pts_mag_sqrd = np.sum(X_train_2nd_pow, axis=1)
        
        # find the dot product of the train and test matrices
        X_times_Xtrain = X.dot(self.X_train.T)
        
        # with everything we have calculated, now we can treat each of these 
        # as the foil of (a-b)^2=a^2-2ab-b^2    
        foil=X_pts_mag_sqrd -2*X_times_Xtrain + X_train_pts_mag_sqrd    
        
        # find the square root of foil, as we were working with magnitude
        # squared values, and we are done:
        dists=np.sqrt(foil)


        return dists

    def predict_labels(self, dists, k=1):

        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in xrange(num_test):
    
            # TODO:
            # 1. sort the elements in the ith row in the dists matrix, use np.argsort
            dists_i_row_sorted=np.argsort(dists[i,:]) 
            # 2. find the labels for the top k closes training points
            k_closest_labels=self.y_train[dists_i_row_sorted[:k]]   

            # 3. count the number of times each label is repeated for the k closest training points
            # use np.bincount
            arr_w_k_incidences = np.bincount(k_closest_labels)
            # 4. find the most repeating label, or the smaller label to break ties
            # use np.argmax
            y_pred[i]=np.argmax(arr_w_k_incidences)
      
        return y_pred