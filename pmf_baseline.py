import pyro
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# parameters specification
epsilon=50 # Learning rate 
reg_param = 0.01 # Regularization parameter 
momentum=0.8 

epoch=0 # interation starting point
maxepoch=50 # subj to change

# load data # Triplets: {user_id, movie_id, rating} # 6337241 ~ 6.4 million ratings
ratings = pd.read_csv('ratings_real.csv') # need to transform to array when fitting the model
ratings_array = ratings.to_numpy()

train_vec, test_vec = train_test_split(ratings_array, test_size=0.2) # split into train and test

mean_ratings = np.mean(train_vec[:, 2]) # get the mean of the ratings

pairs_train = train_vec.shape[0] # training data length (pairs_tr)
pairs_test = test_vec.shape[0] # test data length (pairs_pr)

num_batches = 10 # Number of batches  # numbatches= 9; 
batch_size = 100000 #batch size
num_anime = ratings.anime_id.nunique() # Number of anime: 9927 (num_m)
num_anime += 1
num_users = ratings.user_id.nunique() # Number of users: 69600 (num_p)
num_users += 1
num_feat = 10 # number of latent features; Rank 10 decomposition (10 is faster, but 30 or higher is better)

w_Item = 0.1*np.random.randn(num_anime, num_feat) # Anime feature vectors (w_Item); normal distribution
w_User = 0.1*np.random.randn(num_users, num_feat) # User feature vectors (w_User); normal distribution
w_Item_inc = np.zeros((num_anime, num_feat)) # anime vector increment (w_Item_inc)
w_User_inc = np.zeros((num_users, num_feat)) # users vector increment (w_User_inc)

rmse_train = []
rmse_test = []

while epoch < maxepoch: 
    epoch += 1 # initialize
    
    shuffled_order = np.arange(train_vec.shape[0])  # array based on number of ratings in train data: train_vec
    np.random.shuffle(shuffled_order) # shuffle it

    for batch in range(num_batches): 
        print('epoch %d batch %d ' % (epoch, batch+1)) # maybe too much to print for each batch
        
        test = np.arange(batch_size * batch, batch_size * (batch + 1))
        batch_idx = np.mod(test, shuffled_order.shape[0]) # index that going to be used in this batch
        
        batch_UserID = np.array(train_vec[shuffled_order[batch_idx], 0], dtype='int32')
        batch_ItemID = np.array(train_vec[shuffled_order[batch_idx], 1], dtype='int32')
        
        batch_data = np.multiply(w_User[batch_UserID, :], 
                                 w_Item[batch_ItemID, :])
        ########## compute prediction ##########
        pred_out = np.sum(batch_data, axis = 1)
        rawErr = pred_out - (train_vec[shuffled_order[batch_idx], 2] - mean_ratings) # Default prediction is the mean rating. 
        
        ########## compute gradients ##########
        dw_Item = np.zeros((num_anime, num_feat)) # gradient matrix of anime
        dw_User = np.zeros((num_users, num_feat)) # gradient matrix of users
        
        Ix_User = 2 * np.multiply(rawErr[:, np.newaxis], w_Item[batch_ItemID, :]) + reg_param * w_User[batch_UserID, :] # users gradient
        Ix_Item = 2 * np.multiply(rawErr[:, np.newaxis], w_User[batch_UserID, :]) + reg_param * (w_Item[batch_ItemID, :]) # anime gradient
        
        for i in range(batch_size): # sum gradients
            dw_Item[batch_ItemID[i], :] += Ix_Item[i, :]
            dw_User[batch_UserID[i], :] += Ix_User[i, :]
        
        ########## update users and anime feature (with momentum)##########
        w_User_inc = momentum * w_User_inc + epsilon * dw_User / batch_size;
        w_User =  w_User - w_User_inc;
        
        w_Item_inc = momentum * w_Item_inc + epsilon * dw_Item / batch_size;
        w_Item =  w_Item - w_Item_inc;
        
        ########## compute prediction after updates ########## (this part could be optimized)
        if batch == num_batches - 1:
            pred_out = np.sum(np.multiply(w_User[np.array(train_vec[:, 0], dtype='int32'), :],
                                          w_Item[np.array(train_vec[:, 1], dtype='int32'), :]), axis=1)
            rawErr = pred_out - (train_vec[:, 2] - mean_ratings)
            obj = np.linalg.norm(rawErr) ** 2 + 0.5 * reg_param * (np.linalg.norm(w_User) ** 2 + np.linalg.norm(w_Item) ** 2)
                
            rmse_train.append(np.sqrt(obj / pairs_train))
                
        ########## Compute test error ########## (this part could be optimized)
        if batch == num_batches - 1:
            pred_out = np.sum(np.multiply(w_User[np.array(test_vec[:, 0], dtype='int32'), :],
                                          w_Item[np.array(test_vec[:, 1], dtype='int32'), :]), axis=1)  
            rawErr = pred_out - (test_vec[:, 2] - mean_ratings)
            rmse_test.append(np.linalg.norm(rawErr) / np.sqrt(pairs_test))
            
        ########## Print info ########## (this part could be optimized)
        if batch == num_batches - 1:
            print('The epoch: %f, Training RMSE: %f, Test RMSE %f' % (epoch, rmse_train[-1], rmse_test[-1]))
        
    # if epoch % 10 == 0: 
        

############ training complete ############
# save the users & anime feature for MCMC #
############ ################# ############


