from Data_imports import *
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from tools import *
from sampling import *
import matplotlib.pyplot as plt
'''Importing the Data'''
path_train_data  = r"Code\Problem_5\Data\supernova.txt"
X_train,Y_train = import_X_and_Y_TRAIN(path_train_data)
N = len(X_train)
p = len(X_train[0])

#---------First Part 
'''Fitting the model and predicting'''
Lin_reg_1  = LinearRegression()
Lin_reg_1.fit(X_train,Y_train)
Y_predicted = Lin_reg_1.predict(X_train)

'''Calculating Training Error and RSS'''
Train_Error = mean_squared_error(Y_train, Y_predicted) # Train error using the MSE 
RSS = rss_error(Y_train, Y_predicted)
RSE = rse_error(RSS,N) #estimate of the standard deviation 
TSS = tss_error(Y_train) 
RR = rr_error(TSS, RSS) #the proportion of the total variability in Y that is explained by our model. A higher R^2 indicates a model that explains the variability of Y better, which suggest a better fit of the model of the data.
F_statistic = f_statistic(TSS,RSS,N,p) #High F-statistic value and a corresponding low p-value suggest that the model explains a significant portion of the variance in the response variable, indicating that at least one of the predictors is significantly related to the response. This leads to the rejection of the null hypothesis in favor of the alternative. ( from the ISLP book )
print("Part One:")
print(f"Train Error : {Train_Error}")
print(f"RSS : {RSS}")
print(f"RSE : {RSE}")
print(f"RR : {RR}")
print(f"F-statistic : {F_statistic}",end="\n\n\n\n")

#------Second Part---------
Lin_reg_2 = LinearRegression()
KFOLD = 10
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

err_T_arr.append(err_T)
Err_Cv_arr.append(Err_Cv)
Err_T_arr.append(Err_T)
print("Part Two:")
print(f"Averaged err_T : {np.mean(err_T_arr)}")
print(f"Averaged Err_CV : {np.mean(Err_Cv_arr)}")
print(f"Averaged Err_T : {np.mean(Err_T_arr)}")
plt.figure(figsize=(10, 6))
plt.scatter(Err_T_arr, Err_Cv_arr, label='Err_T vs. Err_CV')
plt.xlabel('Err_T (True Prediction Error)')
plt.ylabel('Err_CV (Cross-Validation Error)')
plt.title('Training Error vs. Cross-Validation Error')
plt.legend()
plt.show()