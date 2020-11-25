import numpy as np

def rmse(true, pred):
    true = np.array(true)
    pred = np.array(pred)
    return np.sqrt(np.mean((true - pred)**2))

def mae(true, pred):
    true = np.array(true)
    pred = np.array(pred)
    return np.mean(abs(true - pred))
