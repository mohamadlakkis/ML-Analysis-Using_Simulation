from scipy.stats import t
import scipy.stats as s
import numpy.random as nr
import numpy as np
import math
import matplotlib.pyplot as plt
'''
In this question we are interested in  
'''
'''
PART A
We will use the Rejection Region for alpha-level test (upper tail test) which is t>= t_alpha, with df = nx+ny-2 = 43 in this example, so if t is greater than this value we will reject H0 and we will take HA, note that in these rejection we will have sometimes where we reject H0 and H0 is true. So it will contain Type I error also. 
'''

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
PART B
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# DELTA_ARR is of the form [(mu_x,mu_y),(mu_x,mu_y)...], note that Delta = mu_x-mu_y and note that I will fix mu_y and vary mu_x like you suggested

DELTA = []
PAIRS_MU = [(9.1, 10), (9.2, 10), (9.3, 10), (9.4, 10), (9.5, 10), (9.6, 10), (9.7, 10), (9.8, 10), (9.9, 10), (10, 10),
            (10.1, 10), (10.2, 10), (10.3, 10), (10.4, 10), (10.5, 10), (10.6, 10), (10.7, 10), (10.9, 10), (11, 10), (11.1, 10), (11.2, 10), (11.3, 10), (11.4, 10), (11.5, 10)]
for i in PAIRS_MU:
    DELTA.append(i[0]-i[1])


'''
this function will return True if T>t_alpha and false otherwise
# This function takes X and Y data and then check if they are greater than t_alpha for a particular alpha 
'''
alpha = 0.1
r = t.ppf(df=20+25-2, q=1-alpha)


def GREATER_THAN_T_ALPHA(X, Y):
    S_X_SQUARE = np.var(X)
    S_Y_SQUARE = np.var(Y)
    n_x = X.size
    n_y = Y.size
    X_BAR = np.mean(X)
    Y_BAR = np.mean(Y)
    sigma_square_hat = ((n_x-1)*S_X_SQUARE + (n_y-1)*S_Y_SQUARE) / (n_x+n_y-2)
    T = (X_BAR-Y_BAR)/(math.sqrt(sigma_square_hat*((1/n_x)+(1/n_y))))
    if T >= r:
        return True
    else:
        return False


N = 2000
ARR_of_prob = []
for pairs in PAIRS_MU:
    mu_x = pairs[0]
    mu_y = pairs[1]
    count = 0
    for i in range(N):
        # This is nature so it knows SIGMA_SQUARE
        SIGMA_SQUARE = 1
        X = nr.normal(mu_x, SIGMA_SQUARE, 20)
        Y = nr.normal(mu_y, SIGMA_SQUARE, 25)
        # here we will play the role of the researcher
        booll = GREATER_THAN_T_ALPHA(X, Y)
        if booll:
            count += 1
    ARR_of_prob.append(count/N)
plt.plot(DELTA, ARR_of_prob, marker='o')
plt.text(-0.3, 0.8, 'P(type I error)',
         fontsize=12, ha='right')
plt.axvline(x=0, color='b', linestyle='--')
plt.axhline(y=0.1, color='g', linestyle='--')
plt.text(-0.1, 0.12, r'$\alpha$', fontsize=12, ha='right')
plt.xlabel('Delta')
plt.ylabel('Power Function')
plt.show()
plt.close()
'''
Comments: We can see that the graph is correct since also when delta = 0 the curve's y compnent is approx alpha = 0.1, and we can that bfore delta = 0 if we reject H0 this is type I error as descirbed on the graph. and yes this is how I expetected the graph to look like. Note that I took more than 9 values only because I wanted the curve to look smoother. And we can see that when delta tends to infinity it will become prob = 1, this infinity is dependent on our sigma_square that we as nature chose. 
Now I will describe how I tackled this part: 
we first choose 9 pairs (in my case more) then for each pair we will choose sigma_square, and we sample from the true distribution 20 X and 25 Y, here we are nature. Now we quit being the nature and since as researchers we don't know sigma_square we will approximate using the formula given in the problem for sigma_squre_hat and then we compute T and check if  T>= t_alpha, df = 43(which can be computed by our laptops), now we simlate this whole process N times to get probabilities, then we will plot these prob, so in the end we will have  each delta with one prob, and we plot them. Notice that if we changed alpha
'''

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
PART C  
p_val = T >= t_obs given that H0 is true. with t_obs is the one calculated in part b, T~T_df
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

'''
PART D
'''

'''
This function calculates t_obs hat we will use in the t.sf
'''


def T_obs(X, Y):
    S_X_SQUARE = np.var(X)
    S_Y_SQUARE = np.var(Y)
    n_x = X.size
    n_y = Y.size
    X_BAR = np.mean(X)
    Y_BAR = np.mean(Y)
    sigma_square_hat = ((n_x-1)*S_X_SQUARE + (n_y-1)*S_Y_SQUARE) / (n_x+n_y-2)
    T_obs = (X_BAR-Y_BAR)/(math.sqrt(sigma_square_hat*((1/n_x)+(1/n_y))))
    return T_obs


rv = t(df=43)
# This array is of the form [[],[],[]] each index is another array for eahc specific delta
arr = [[]]
count = 0
for pairs in PAIRS_MU:
    mu_x = pairs[0]
    mu_y = pairs[1]
    for i in range(N):
        # This is nature so it knows SIGMA_SQUARE
        SIGMA_SQUARE = 1
        X = nr.normal(mu_x, SIGMA_SQUARE, 20)
        Y = nr.normal(mu_y, SIGMA_SQUARE, 25)
        # Observer
        tobs = T_obs(X, Y)
        arr[count].append(rv.sf(tobs))
    # this count just to append the correct sub-array for the paritcular delta we are working with
    count += 1
    arr.append([])
# PLotting
for a in range(len(arr)-1):
    # plotting eahc hist for each p-values of parituclar delta
    plt.hist(arr[a], density=True)
    plt.xlabel("P-values")
    plt.ylabel("DENSITY")
    m = str(PAIRS_MU[a]) + r'$\delta$'
    plt.title(label=m)
    # plt.show() #in oorder to see each graph separetly, I did this before plotting them all on one figure, but you told me that you need them on one
plt.show()
'''
We can notice that if delta = 0, p_value ~ unfiorm[0,1]
We can notice that when delta becomes larger and larger the desnity on low values of p is larger, whcih is consistent with the data in part B since when delta is large enough the prob to reject H0 tends to 1 as delta tends to infinity. 
We also can see that when delta becomes smaller and smaller the p values'sdensty on high prob becomes larger which is consitnet with part B because in part B prob of rejecting H0 when delta is so small tends to 0 
'''
