# -*- coding: utf-8 -*-
import numpy as np
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer.mcmc import MCMC, NUTS


class PMF(): 
    def __init__(self, n_user, n_anime, n_feature, num_batches=10, batch_size=1000, epsilon=50.0, momentum=0.8, seed=None, regu_param=0.01, max_rating=None, min_rating=None):

        super(PMF, self).__init__()
        self.n_user = n_user # 69600
        self.n_anime = n_anime # 9927
        self.n_feature = n_feature # number of features for decompisition

        self.random_state = RandomState(seed)

        self.num_batches = num_batches # number of batches
        self.batch_size = batch_size # batch size, default 1000

        self.epsilon = float(epsilon) # learning rate
        self.regu_param = regu_param # regularization parameter
        self.momentum = float(momentum)

        # data state
        self.mean_rating_ = None
        # user/anime features vectors
        self.user_features_ = 0.1 * self.random_state.rand(n_user, n_feature)
        self.item_features_ = 0.1 * self.random_state.rand(n_anime, n_feature)

        self.rmse_train = []
        self.rmse_test = []

    def fit(self, train_in, test_in):

        self.mean_rating_ = np.mean(train_in[:, 2])

        # momentum
        u_feature_mom = np.zeros((self.n_user, self.n_feature))
        i_feature_mom = np.zeros((self.n_anime, self.n_feature))
        # gradient
        u_feature_grads = np.zeros((self.n_user, self.n_feature))
        i_feature_grads = np.zeros((self.n_anime, self.n_feature))
        for iteration in xrange(n_iters):
            logger.debug("iteration %d...", iteration)

            self.random_state.shuffle(ratings)

            for batch in xrange(batch_num):
                start_idx = int(batch * self.batch_size)
                end_idx = int((batch + 1) * self.batch_size)
                data = ratings[start_idx:end_idx]

                # compute gradient
                u_features = self.user_features_.take(
                    data.take(0, axis=1), axis=0)
                i_features = self.item_features_.take(
                    data.take(1, axis=1), axis=0)
                preds = np.sum(u_features * i_features, 1)
                errs = preds - (data.take(2, axis=1) - self.mean_rating_)
                err_mat = np.tile(2 * errs, (self.n_feature, 1)).T
                u_grads = i_features * err_mat + self.reg * u_features
                i_grads = u_features * err_mat + self.reg * i_features

                u_feature_grads.fill(0.0)
                i_feature_grads.fill(0.0)
                for i in xrange(data.shape[0]):
                    row = data.take(i, axis=0)
                    u_feature_grads[row[0], :] += u_grads.take(i, axis=0)
                    i_feature_grads[row[1], :] += i_grads.take(i, axis=0)

                # update momentum
                u_feature_mom = (self.momentum * u_feature_mom) + \
                    ((self.epsilon / data.shape[0]) * u_feature_grads)
                i_feature_mom = (self.momentum * i_feature_mom) + \
                    ((self.epsilon / data.shape[0]) * i_feature_grads)

                # update latent variables
                self.user_features_ -= u_feature_mom
                self.item_features_ -= i_feature_mom

            # compute RMSE
            train_preds = self.predict(ratings[:, :2])
            train_rmse = RMSE(train_preds, ratings[:, 2])
            logger.info("iter: %d, train RMSE: %.6f", iteration, train_rmse)

            # stop when converge
            if last_rmse and abs(train_rmse - last_rmse) < self.converge:
                logger.info('converges at iteration %d. stop.', iteration)
                break
            else:
                last_rmse = train_rmse
        return self

    def predict(self, data):

        if not self.mean_rating_:
            raise NotFittedError()

        u_features = self.user_features_.take(data.take(0, axis=1), axis=0)
        i_features = self.item_features_.take(data.take(1, axis=1), axis=0)
        preds = np.sum(u_features * i_features, 1) + self.mean_rating_

        if self.max_rating:
            preds[preds > self.max_rating] = self.max_rating

        if self.min_rating:
            preds[preds < self.min_rating] = self.min_rating
        return preds
