# Final Project Repository

This repository is duplicated from https://github.com/akucukelbir/probprog-finalproject

For course [Machine Learning with Probabilistic Programming](https://www.proditus.com/mlpp2020)

## Project Summary

The goal of the project is to perform probabilistic matrix factorization on the anime dataset: to identify the potential interest of users and recommend to them new anime. 

## Data Source

The data is from the [Anime Recommendations Database](https://www.kaggle.com/CooperUnion/anime-recommendations-database)

- The `rating.csv` contains ratings that users give to different anime
  - `user_id`
  - `anime_id`
  - `rating`
- The `anime.csv` contains detailed information about anime

## Development

Python 3.8.5

First activate your virtual environment and install all dependencies. 

```
$ workon venv
(venv)$ pip install -r requirements.txt
```

To run final-notebook

```
(venv)$ jupyter notebook
```

To run future work script

```
$ cd future-work
(venv)$ python pmf_semi_pyro_largescale.py
```

All python scripts have passed `flake8` linting. 

