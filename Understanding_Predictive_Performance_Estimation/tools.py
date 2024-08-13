from sklearn.model_selection import cross_val_score
import numpy as np
from sampling import *
from sklearn.metrics import mean_squared_error
def rss_error(Y_train, Y_predicted):
    residual = Y_train - Y_predicted
    return np.sum(residual ** 2)
def rse_error(RSS,n):
    return  (RSS / (n-2) )**0.5
def tss_error(Y_train):
    MEAN = np.mean(Y_train)
    return np.sum((Y_train -MEAN) ** 2) 
def rr_error(TSS, RSS):
    return 1 -  (RSS / TSS) 
def f_statistic(TSS,RSS,n,p):
    regSS = TSS-RSS #This difference is the amount of variability in Y that the model explains. 
    return (regSS/p)/(RSS/(n-p-1))
def compute_CV_Error(model,X,Y,KFOLD):
    #Note that: To align with sckit-learn's convention of "higher is better" for scoring functions, the MSE is negated, (since in general higher is worse while dealing with errors, and with sckit learn they usually deal with accuracy !)
    scores = cross_val_score(model, X, Y, scoring='neg_mean_squared_error', cv=KFOLD)
    Err_Cv = -scores.mean() # Higher is not better!!
    return Err_Cv
def True_Pred_Error_Given_P_hat_and_T(Y_train,X_train,model):
    '''This function computes the True Err(T) with the assumption that the joint distribution of X,Y is known as the sample_P_hat_X_Y , with a given model that is trained on a sample from this distribution'''
    Y_pred = model.predict(X_train) 
    return mean_squared_error(Y_pred, Y_train)

def experiment_2(X_train,Y_train,Lin_reg_2,KFOLD):
    N  = len(X_train)
    err_T_arr = []
    Err_Cv_arr = []
    Err_T_arr = []

    for i in range(1000):
        X_bootstrapped_train,Y_bootstrapped_train = sample_P_hat_X_Y(N,X_train, Y_train)
        Lin_reg_2.fit(X_bootstrapped_train, Y_bootstrapped_train)
        Y_predicted_2 = Lin_reg_2.predict(X_bootstrapped_train)

        err_T = mean_squared_error(Y_bootstrapped_train, Y_predicted_2) #Train error using the MSE !
        Err_Cv = compute_CV_Error(Lin_reg_2, X_bootstrapped_train, Y_bootstrapped_train, KFOLD) 
        Err_T  = True_Pred_Error_Given_P_hat_and_T(Y_train,X_train,Lin_reg_2) #Notice that this is very similar to the training error, but they are not the same here we are taking all possibilities from the P_hat distribution where as in the training error ( because it is with repetition we might not include all values !), notice that here we trained the model on the bootsrapped sample and know we will get this error by taking all possible values of the joint distribution X,Y which is in our assumption known 
        err_T_arr.append(err_T)
        Err_Cv_arr.append(Err_Cv)
        Err_T_arr.append(Err_T)
    err_T_arr = np.array(err_T_arr)
    Err_Cv_arr = np.array(Err_Cv_arr)
    Err_T_arr = np.array(Err_T_arr)
    return err_T_arr, Err_Cv_arr, Err_T_arr