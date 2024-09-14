import numpy as np

def softmax(x : np.ndarray) -> np.ndarray:
    """
    Compute the softmax of vector x.
    """
    if x.ndim == 1:
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
    elif x.ndim == 2:
        e_x = np.exp(x - np.max(x, axis=1)[:, None])
        return e_x / e_x.sum(axis=1)[:, None]
    else:
        raise ValueError("Invalid input dimension.")

def layer_norm(x : np.ndarray, eps : float = 1e-5) -> np.ndarray:
    """
    Layer normalization.
    """
    if x.ndim == 1:
        mean = np.mean(x)
        std = np.std(x)
        return (x - mean) / (std + eps)
    elif x.ndim == 2:
        mean = np.mean(x, axis=1, keepdims=True)
        std = np.std(x, axis=1, keepdims=True)
        return (x - mean) / (std + eps)
    else:
        raise ValueError("Invalid input dimension.")

def ReLU(x : np.ndarray) -> np.ndarray:
    """
    Rectified Linear Unit.
    """
    return np.maximum(0, x)

def padding_matrix(x : int, y : int, l : int) -> np.ndarray:
    """
    Padding matrix x to max_length.
    """
    mask_matrix = np.zeros((x, y))
    for i in range(x):
        if i > l - 1 and i <= x - 1:
            mask_matrix[i] = -np.inf
    return mask_matrix

def position_mask(x : int, y : int, step : int) -> np.ndarray:
    """
    Position mask.
    """
    mask_matrix = np.zeros((x, y))
    for i in range(x):
        for j in range(y):
            if i < j - step:
                mask_matrix[i][j] = -np.inf
    return mask_matrix