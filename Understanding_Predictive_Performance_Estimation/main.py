from Data_imports import *
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from tools import *
from sampling import *
import matplotlib.pyplot as plt
'''Importing the Data'''
path_train_data  = r"Problem_5\Data\supernova.txt"
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
print(f"F-statistic : {F_statistic}",end="\n\n")

#------Second Part---------

Lin_reg_2 = LinearRegression()
KFOLD = 10
#Please see experiment_2 very interesting analysis there as well !
err_T_arr,Err_Cv_arr,Err_T_arr = experiment_2(X_train,Y_train,Lin_reg_2,KFOLD)


'''Please find the analysis in the bottom of this page (the commenting section)'''

print("Part Two:")
print(f"Averaged err_T : {np.mean(err_T_arr)}")
print(f"Averaged Err_CV : {np.mean(Err_Cv_arr)}")
print(f"Averaged Err_T : {np.mean(Err_T_arr)}")
plt.figure(figsize=(10, 6))
plt.scatter(Err_T_arr, Err_Cv_arr, label='Err_T vs. Err_CV')
plt.xlabel('Err_T (True Prediction Error)')
plt.ylabel('Err_CV (Cross-Validation Error)')
plt.title('Training Error vs. Cross-Validation Error')
plt.grid(True)
plt.legend()
plt.show()
#-------analysis-------
'''
The results validate the lemma proposed that the CV estimate of error is estimating the Err and not the Err(T), because if it were the case that CV estimates the Err(T), then we would have expected to find many points on the y=x line, but as we see that we have many points not in that region, and then we see that indeed if we take the average over many training sets we will get ourselves a prediction that is so close to Err, and so indeed the CV estimate of error on average estimate the Err and not the Err(T), since if we take E_T(Err(T)) this becomes Err, so as we see in the averaged results AVG Err_T is very close to average Err_CV. And in order to validate this even further, I have repeated the experiment many times to see what will the plot of the resulted averages look like, and I saw that they are all around (if not on) the point of the True Err, which lies on the line y = x ! AMAZING RESULTS ! ( Note that I then discarded the code since it is trivial to implement it just wrap the experiment_2 function in a loop and after that add the average of each iteration to an array and plot that array )
'''