import numpy as np

def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    
    loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
    return loss