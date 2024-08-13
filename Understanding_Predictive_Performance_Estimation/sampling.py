import numpy as np
def sample_P_hat_X_Y(sample_size,X_values,Y_values):
    X_OUTPUT = []
    Y_OUTPUT = []
    for i in range(sample_size):
        #which guarantee the "with replacement" condition
        j = np.random.randint(0, len(X_values))
        X_OUTPUT.append(X_values[j])
        Y_OUTPUT.append(Y_values[j])
    X_OUTPUT = np.array(X_OUTPUT)
    Y_OUTPUT = np.array(Y_OUTPUT)
    return X_OUTPUT, Y_OUTPUT