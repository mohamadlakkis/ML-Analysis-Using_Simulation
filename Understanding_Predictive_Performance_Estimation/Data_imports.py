import numpy as np
def import_X_and_Y_TRAIN(filename):
    data = open(filename, 'r')
    i = 0
    X_train = []
    Y_train = []
    for line in data:
        if i==0:
            i+=1

        else:
            temp = line.split(' ')[1:]
            temp_2 = float(temp[-1])
            temp = temp[:-1]
            temp = [float(x) for x in temp]
            X_train.append(temp)
            Y_train.append(temp_2)
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    return X_train,Y_train