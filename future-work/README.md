# About

This directory is for future work as mentioned before in our final-notebook

## Data

the data is still the Anime dataset from [Kaggle](https://www.kaggle.com/CooperUnion/anime-recommendations-database), but we recommend exclude all "-1" ratings before running the program. 

## Plan

- `pmf_semi_pyro_largescale.py`: compute MAP solution on a large scale data. It is mostly based on pytorch. We are working on having a pyro version of it. 
- Use the MAP solution to initialize MCMC inference with pyro. 
- Build classification model based on estimation of latent feature. 

# Acknowledgement

The code in python is largely based on the matlab code provided by Ruslan Salakhutdinov. 

They are available on http://www.cs.toronto.edu/~rsalakhu/BPMF.html