import numpy as np

def calculate_eigenvalues(matrix: list) -> np.ndarray:
    """
    Returns a sorted NumPy array of real eigenvalues.
    """
    matrix = np.array(matrix)
    val, vec = np.linalg.eig(matrix)
    val.sort()
    return val