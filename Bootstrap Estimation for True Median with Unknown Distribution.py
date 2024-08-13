from scipy.stats import t
import scipy.stats as s
import numpy.random as nr
import numpy as np
import math
import matplotlib.pyplot as plt
d = np.array([0.50387383, 0.72883327, 0.80320775, 0.69149048, 0.46768236, 1.17713452, 1.03276632, 1.54509969, 1.18642456, 1.7771954, 1.88873037, 1.0888725, 1.2032791, 1.09910349, 1.04986028, 2.49231659, 1.98631131, 1.12588689, 6.00987384, 1.72916505, 3.58359318, 1.70768768, 1.07343242, 1.45174748, 1.3107978,
             1.54092739, 1.02307098, 2.35969702, 2.44963857, 1.04177858, 1.20644624, 1.28461754, 1.50839907, 2.18319367, 1.17165567, 1.12682264, 1.13097293, 1.08454239, 1.14149616, 1.19055332, 1.21763137, 1.29158482, 1.45633721, 1.36990089, 1.90800037, 1.92686206, 1.97682805, 1.64529231, 1.6621115, 4.77535181])
theta_hat = np.median(d)

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
PART A 
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

'''
Step 1: CREATING THE SAMPLER FUNCTIONS
'''

# NON PARAMETRIC BOOTSTRAP SAMPLER


def NON_FHAT_SAMPLER(size=50):
    dstar = np.random.choice(d, size, replace=True)
    return dstar


# PARAMETRIC BOOTSTRAP SAMPLER

# Calculating lambda_MLE
q = np.sum(d >= 1)
total = np.sum(np.log(d[d >= 1]))
lambda_MLE = q/total


def PARA_FHAT_SAMPLER(N=50, v=0.2, lamb=lambda_MLE):
    u = nr.uniform(0, 1, N)
    u1 = u[u < v]
    X1 = ((u1/v)**0.5)[u1 < 1000]
    u2 = u[(u >= v)]
    X2 = (((v-1)/(u2-1))**(1/lamb))[u2 > -999]
    X = np.concatenate((X1, X2))
    return X


'''
Step 2: CREATING THE STANDARD ERROR FUNCTION for NON and for PARA
This function will return the bootstrap Estimate of the STANDARD ERROR of theta_hat
'''

# A PREPARATION STEP, a function that generates B replicates of the median


def B_Replicates(B=10000, PARA=False):
    replicates = np.ones(B)
    # this means we will be sampling from non_para
    if PARA == False:
        for i in range(B):
            # the sampler by deafult will give us 50 values
            X = NON_FHAT_SAMPLER()
            replicates[i] = np.median(X)
    else:
        # This means we will be sampling from the para
        for i in range(B):
            # The sampler by default will give us 50 values
            X = PARA_FHAT_SAMPLER()
            replicates[i] = np.median(X)
    return replicates


def STANDARD_ERROR(replicates):
    var_theta_hat = np.var(replicates)
    return math.sqrt(var_theta_hat)


'''
Step 3: CREATE the first version of CI (the 100(1-alpha)%), it will take by dafault alpha_over_2 = 0.025 (the 2.5%)
'''


def Version_1_CI(replicates, alpha_over_2=0.025):
    separation_value_1 = np.percentile(replicates, 100 * alpha_over_2)
    separation_value_2 = np.percentile(replicates, 100 * (1-alpha_over_2))
    return (separation_value_1, separation_value_2)


'''
Step 4: CREATE the second version of CI the normal one 
'''


def Version_2_CI(replicates, alpha_over_2=0.025):
    rv = s.norm(loc=0, scale=1)
    z = rv.ppf(1-alpha_over_2)
    std_error = STANDARD_ERROR(replicates)
    error = std_error*z
    left_bound = theta_hat-error
    right_bound = theta_hat+error
    return (left_bound, right_bound)


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
PART B
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

'''
Step 1: Drawing the 2 hist of the replicates
'''

replications_NON = B_Replicates(PARA=False, B=10000)
replications_PARA = B_Replicates(PARA=True, B=10000)
plt.hist(replications_NON, density=True, bins=30)
plt.title("Replicates from the Non Parametric Distribution")
plt.show()
plt.close()
plt.hist(replications_PARA, density=True, bins=30)
plt.title("Replicates from the Parametric Distribution")
plt.show()
plt.close()

'''
Step 2:Drawing the pdf of the true sampling distribution by simulation 
'''
iter = 1000000
median_simulation = np.ones(iter)
for i in range(iter):
    X = PARA_FHAT_SAMPLER(N=50, v=0.2, lamb=2)
    median_simulation[i] = np.median(X)
plt.hist(median_simulation, density=True, bins=30)
plt.show()

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
PART C
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''
Step 1: Based on the HIST in part B
'''
fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(median_simulation, bins=30, alpha=0.5,
        label='Median Simulation', density=True)
ax.hist(replications_NON, bins=30, alpha=0.5,
        label='Non-Parametric', density=True)
ax.hist(replications_PARA, bins=30, alpha=0.5,
        label='Parametric', density=True)
ax.set_xlabel('Value')
ax.set_ylabel('Density')
ax.legend()
plt.show()
# we can clearly see that the FHAT parametric is more accurate, which makes sense since in this case we know more paramteres of the true distibution, we know Thr true F which the one in problem 1 and we know nu. So logically it will make more sense for this distribution to make more sense (IN THIS PROBLEM )


'''
Step 2: using CI
Plan: First calculate the true median that we have. Second Generate Confidence Inteverlas from PARA and NON and see how many of them contain the true median, whihc like we discussed it is more likely to contain the confidence intervals, because we are generating based on 1 sample (d), but if we generate more samples (like d from the true disributuion we will see that PARA CIs is better than NON CIs since we know more info about the distirution so we lower the error and extreme values, because the non para will give weights all points the same weight which may cause lots of errors since it is vulnerable for the tails (i.e extreme values))
'''


# Approximating the True median of the true distibution by simulation for NON_para and PARA
iter = 200
# we will use the expected value to approximate the true median
'''
#Computing The median by simultion 
expected_median_NON = 0
expected_median_PARA = 0
for i in range(iter):
    X1 = NON_FHAT_SAMPLER()
    X2 = PARA_FHAT_SAMPLER()
    expected_median_NON += np.median(X1)
    expected_median_PARA += np.median(X2)
expected_median_NON = expected_median_NON/iter
expected_median_PARA = expected_median_PARA/iter
# Note that i calculated the true median
'''
# computing the True MEDIAN
'''
we do so by solving for F(X) = 0.5 from the problem 1 distribution the second one we take. 
Total median = M2*w2 , where  M2 median second dsitrubtion sice nu (v) is less than 0.5
'''

v = 0.2
lamb = 2
TRUE_MEDIAN = ((1-v)/(0.5))**(1/lamb)
print(TRUE_MEDIAN)
# this values makes sense because if we look on the hist we will see that approx the median is on the point 1.27~2.26
# Drawing the median on the hist to see it
fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(median_simulation, bins=30, alpha=0.5,
        label='Median Simulation', density=True)
ax.hist(replications_NON, bins=30, alpha=0.5,
        label='Non-Parametric', density=True)
ax.hist(replications_PARA, bins=30, alpha=0.5,
        label='Parametric', density=True)
ax.set_xlabel('Value')
ax.set_ylabel('Density')
ax.legend()
ax.axvline(x=TRUE_MEDIAN, color='red', linestyle='--', label='True Median')
plt.show()
# Generating Confidence Intervals for NON

iter = 100
vers1_NON = 0
vers2_NON = 0
vers1_PARA = 0
vers2_PARA = 0
for i in range(iter):
    # NON
    replicates_NON = B_Replicates()
    low1_NON, high1_NON = Version_1_CI(replicates_NON)
    low2_NON, high2_NON = Version_2_CI(replicates_NON)
    if low1_NON <= TRUE_MEDIAN <= high1_NON:
        vers1_NON += 1
    if low2_NON <= TRUE_MEDIAN <= high2_NON:
        vers2_NON += 1
    # PARA
    replicates_PARA = B_Replicates(PARA=True)
    low1_PARA, high1_PARA = Version_1_CI(replicates_PARA)
    low2_PARA, high2_PARA = Version_2_CI(replicates_PARA)
    if low1_PARA <= TRUE_MEDIAN <= high1_PARA:
        vers1_PARA += 1
    if low2_PARA <= TRUE_MEDIAN <= high2_PARA:
        vers2_PARA += 1
# These calculations will be prob 1 (like discusssed earlier )
print("Probabilities related to the confidence interval")
print("NON-Parametric:")
print("prob that the true median is in the confidence interval(Version 1) for NON Parametric = ", vers1_NON/iter)
print("prob that the true median is in the confidence interval(Version 2) for NON Parametric = ", vers2_NON/iter)
print("P")
print(" ")
print("Parametric:")
print("prob that the true median is in the confidence interval(Version 1) for Parametric = ", vers1_PARA/iter)
print("prob that the true median is in the confidence interval(Version 2) for Parametric = ", vers2_PARA/iter)