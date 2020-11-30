# The script is modified from the matlab code provided by Ruslan Salakhutdinov
# http://www.cs.toronto.edu/~rsalakhu/BPMF.html

import numpy as np

import pandas as pd

import pyro

from sklearn.model_selection import train_test_split

import torch
# from torch.distributions import constraints

# import pyro.distributions as dist
# import pyro.optim as optim

# from pyro.infer import EmpiricalMarginal, SVI, Trace_ELBO, Predictive
# from pyro.contrib.autoguide import AutoMultivariateNormal
# from pyro.infer.mcmc.api import MCMC
# from pyro.infer.mcmc import NUTS

pyro.set_rng_seed(1)
assert pyro.__version__.startswith('1.4.0')
np.random.seed(seed=1)

# parameters specification
epsilon = 50  # Learning rate
reg_param = 0.01  # Regularization parameter
momentum = 0.8

epoch = 0  # interation starting point
maxepoch = 50  # subj to change

# load data # Triplets: {user_id, movie_id, rating}
# 6337241 ~ 6.4 million ratings
# need to transform to array when fitting the model
# for data, go the the kaggle website and download ratings.csv;
# and exclude all "-1" ratings
ratings = pd.read_csv('ratings_real.csv')
ratings_array = ratings.to_numpy()

train_vec, test_vec = train_test_split(
    ratings_array, test_size=0.2)  # split into train and test

mean_ratings = np.mean(train_vec[:, 2])  # get the mean of the ratings

pairs_train = train_vec.shape[0]  # training data length (pairs_tr)
pairs_test = test_vec.shape[0]  # test data length (pairs_pr)

num_batches = 30  # Number of batches
batch_size = 100000  # batch size
num_anime = ratings.anime_id.nunique()  # Number of anime: 9927 (num_m)
num_users = ratings.user_id.nunique()  # Number of users: 69600 (num_p)
# number of latent features; Rank 10 decomposition (10 is faster
# but 30 is good)
num_feat = 30
uf_mean0 = torch.zeros([num_users, num_feat])
uf_std0 = torch.ones([num_users, num_feat])
# Anime
af_mean0 = torch.zeros([num_anime, num_feat])
af_std0 = torch.ones([num_anime, num_feat])

w_User = 0.1 * (pyro.sample("s",
                            pyro.distributions.Normal(
                                loc=uf_mean0,
                                scale=uf_std0).to_event(2)).numpy())
w_Item = 0.1 * (pyro.sample("e",
                            pyro.distributions.Normal(
                                loc=af_mean0,
                                scale=af_std0).to_event(2)).numpy())
# w_Item = 0.1*np.random.randn(num_anime, num_feat)
# Anime feature vectors (w_Item); normal distribution
# w_User = 0.1*np.random.randn(num_users, num_feat)
# User feature vectors (w_User); normal distribution
# anime vector increment (w_Item_inc)
w_Item_inc = np.zeros((num_anime, num_feat))
# users vector increment (w_User_inc)
w_User_inc = np.zeros((num_users, num_feat))

rmse_train = []
rmse_test = []

while epoch < maxepoch:
    epoch += 1  # initialize

    # array based on number of ratings in train data: train_vec
    shuffled_order = np.arange(train_vec.shape[0])
    np.random.shuffle(shuffled_order)  # shuffle it

    for batch in np.arange(num_batches):
        # print('epoch %d batch %d ' % (epoch, batch+1))

        test = np.arange(batch_size * batch, batch_size * (batch + 1))
        # index that going to be used in this batch
        batch_idx = np.mod(test, shuffled_order.shape[0])

        batch_UserID = np.array(
            train_vec[shuffled_order[batch_idx], 0], dtype='int32')
        batch_ItemID = np.array(
            train_vec[shuffled_order[batch_idx], 1], dtype='int32')

        # need to minus one as index starts from 0
        batch_data = np.multiply(w_User[batch_UserID - 1, :],
                                 w_Item[batch_ItemID - 1, :])

        # compute prediction
        pred_out = np.sum(batch_data, axis=1)
        # Default prediction is the mean rating.
        rawErr = pred_out - \
            (train_vec[shuffled_order[batch_idx], 2] - mean_ratings)

        # compute gradients
        dw_Item = np.zeros((num_anime, num_feat))  # gradient matrix of anime
        dw_User = np.zeros((num_users, num_feat))  # gradient matrix of users

        # users gradient
        Ix_User = 2 * np.multiply(rawErr[:, np.newaxis],
                                  w_Item[batch_ItemID - 1, :]) + \
            reg_param * (w_User[batch_UserID - 1, :])
        # anime gradient
        Ix_Item = 2 * np.multiply(rawErr[:, np.newaxis],
                                  w_User[batch_UserID - 1, :]) + \
            reg_param * (w_Item[batch_ItemID - 1, :])

        for i in range(batch_size):  # sum gradients
            dw_Item[batch_ItemID[i] - 1, :] += Ix_Item[i, :]
            dw_User[batch_UserID[i] - 1, :] += Ix_User[i, :]

        # update users and anime feature with momentum
        w_User_inc = momentum * w_User_inc + epsilon * dw_User / batch_size
        w_User = w_User - w_User_inc

        w_Item_inc = momentum * w_Item_inc + epsilon * dw_Item / batch_size
        w_Item = w_Item - w_Item_inc

        # compute prediction after updates (this part could be optimized)
        if batch == num_batches - 1:
            pred_out = np.sum(np.multiply(
                w_User[np.array(train_vec[:, 0] - 1, dtype='int32'), :],
                w_Item[np.array(train_vec[:, 1] - 1, dtype='int32'), :]),
                axis=1)
            rawErr = pred_out - (train_vec[:, 2] - mean_ratings)
            obj = np.linalg.norm(rawErr) ** 2 + 0.5 * reg_param * \
                (np.linalg.norm(w_User) ** 2 + np.linalg.norm(w_Item) ** 2)

            rmse_train.append(np.sqrt(obj / pairs_train))

        # Compute test error (this part could be optimized)
        if batch == num_batches - 1:
            pred_out = np.sum(np.multiply(
                w_User[np.array(test_vec[:, 0] - 1, dtype='int32'), :],
                w_Item[np.array(test_vec[:, 1] - 1, dtype='int32'), :]),
                axis=1)
            rawErr = pred_out - (test_vec[:, 2] - mean_ratings)
            rmse_test.append(np.linalg.norm(rawErr) / np.sqrt(pairs_test))

        # Print info (this part could be optimized)
        if batch == num_batches - 1:
            print('The epoch: %d, Training RMSE: %f, Test RMSE %f' %
                  (epoch, rmse_train[-1], rmse_test[-1]))

    # if epoch % 10 == 0:


# training complete, save the users & anime feature for MCMC

# np.savetxt('./pmf_weights/users_feature.csv', w_User, delimiter=',')
# np.savetxt('./pmf_weights/anime_feature.csv', w_Item, delimiter=',')
