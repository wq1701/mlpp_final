#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 22:30:21 2020

@author: weiwei_qi
"""

# toy example from https://beckernick.github.io/matrix-factorization-recommender/

import pandas as pd
import numpy as np

ratings_df = pd.read_csv("ratings_top100_users.csv")

ratings_df_matrix = ratings_df.pivot(index = 'user_id', columns = 'anime_id', values = 'rating').fillna(0)

ratings_df_legit = ratings_df[ratings_df["rating"] != -1]

ratings_df_legit_matrix = ratings_df_legit.pivot(index = 'user_id', columns = 'anime_id', values = 'rating').fillna(0)

# ratings_df_legit_matrix.head()

R = ratings_df_legit_matrix

user_ratings_mean = np.mean(R, axis = 1)

R_demeaned = R - user_ratings_mean.reshape(-1, 1)

from scipy.sparse.linalg import svds
U, sigma, Vt = svds(R_demeaned, k = 50)

sigma = np.diag(sigma)