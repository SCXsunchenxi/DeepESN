import numpy as np

# NarmaX serie
def create_narmaX_data(t):
    """
    Create a new input data given the length and the corresponding
    narmaX sequence.
    """
    u = np.random.uniform(0,0.5,t) # Input values
    # NARMA sequence
    y = np.zeros(t)
    for i in range(t-1):
        sum_t = np.sum(y[0:i]) if i <= 9 else np.sum(y[i-9:i+1])
        if i <= 9:
            y[i+1] = 0.3 * y[i] + 0.05 * sum_t * y[i] + 0.1
        else:
            y[i+1] = 0.3 * y[i] + 0.05 * sum_t * y[i] + 1.5 * u[i] * u[i - 9] + 0.1
    u = u[np.newaxis,20:]
    y = y[np.newaxis,20:]
    return u, y