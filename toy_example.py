#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 22:56:11 2020

@author: weiwei_qi
"""

import pandas as pd
import numpy as np

ratings_list = [i.strip().split("::") for i in open('/Users/weiwei_qi/wd/mlpp_final/toy_data/ratings.dat', 'r').readlines()]
users_list = [i.strip().split("::") for i in open('/Users/weiwei_qi/wd/mlpp_final/toy_data/users.dat', 'r').readlines()]
movies_list = [i.strip().split("::") for i in open('/Users/weiwei_qi/wd/mlpp_final/toy_data/movies.dat', 'r', encoding = "ISO-8859-1").readlines()]

ratings = np.array(ratings_list)
users = np.array(users_list)
movies = np.array(movies_list)

ratings_df = pd.DataFrame(ratings_list, columns = ['UserID', 'MovieID', 'Rating', 'Timestamp'], dtype = int)
movies_df = pd.DataFrame(movies_list, columns = ['MovieID', 'Title', 'Genres'])
movies_df['MovieID'] = movies_df['MovieID'].apply(pd.to_numeric)

movies_df.head()

R_df = ratings_df.pivot(index = 'UserID', columns ='MovieID', values = 'Rating').fillna(0)
R_df.head()

R = R_df.as_matrix()
user_ratings_mean = np.mean(R, axis = 1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)