import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    # Write code here
    w = np.zeros(X.shape[1:])
    b = 0.0
    for i in range(steps):
        z = X @ w + b
        y_cap = _sigmoid(z)
        loss = -np.mean((y *  np.log(y_cap) + (1 - y) * np.log(1 - y_cap)))
        error = y_cap - y
        grad_w = X.T @ error * (1/X.shape[0])
        grad_b = np.mean(error)
        w = w - (lr * grad_w)
        b = b - (lr * grad_b)
    return w, b