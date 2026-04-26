import numpy as np 

def t_div(Y_pred, Y_true, epsilon=1.0, dt=0.01):

    distances = np.linalg.norm(Y_pred - Y_true, axis=1)
    exceeds = np.where(distances > epsilon)[0]
    
    if len(exceeds) == 0:
        return len(Y_pred) * dt
    
    t_div = exceeds[0] * dt
    print(f"Divergence time: {float(t_div)}")
    return t_div
