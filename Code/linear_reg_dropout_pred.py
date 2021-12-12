
#%% ME760 Project - Predicting dropout rates from state-wise education data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import norm
from scipy.stats.distributions import chi2
import seaborn as sn

import math

#%%
def read_data(filename):
    df = pd.read_csv(filename)
    df['Bias'] = 1;
    df = df.drop(columns=['SL. No.'])
    return df

#%%
def linear_regression(my_xvector,df):
    print('Linear regression using own code:')
    y = df['SB_Dropout'].values
    X = df[['Bias','SB_ER','Sec_Only_Comp','SB_Toilet','Sec_Only_Elec','Sec_Only_Water']].values
    theta_hat = np.matmul(np.linalg.inv(np.matmul(X.T,X)),np.matmul(X.T,y));
    print('Coefficients:',theta_hat)
    
    N = len(y);
    my_xvector = np.insert(my_xvector, [0], 1);
    y_pred = np.matmul(my_xvector.T, theta_hat);
    print('predicted dropout rate:',y_pred)
    var_hat = (1.0/N)*np.matmul((y - np.matmul(X,theta_hat)).T,(y - np.matmul(X,theta_hat)));
    # print('var_hat',var_hat)
    temp = var_hat*np.matmul(my_xvector.T,np.matmul(np.linalg.inv(np.matmul(X.T,X)),my_xvector));
    tau = norm.ppf(alpha/2,loc = 0,scale = math.sqrt(temp.item(0)))
    # print('tau',tau)
    print('confidence interval:',(y_pred+tau),(y_pred-tau))
    return y_pred
#%%
def LR_significance(my_xvector,df,alpha):
    y = df['SB_Dropout'].values
    Col_names = ['Bias','SB_ER','Sec_Only_Comp','SB_Toilet','Sec_Only_Elec','Sec_Only_Water']
    X = df[Col_names].values

    theta_hat = np.matmul(np.linalg.inv(np.matmul(X.T,X)),np.matmul(X.T,y));
    N = len(y);
    my_xvector = np.insert(my_xvector, [0], 1);
    y_pred = np.matmul(my_xvector.T, theta_hat);

    var_hat = (1.0/N)*np.matmul((y - np.matmul(X,theta_hat)).T,(y - np.matmul(X,theta_hat)));
    # print('var_hat',var_hat)
    temp = var_hat*np.matmul(my_xvector.T,np.matmul(np.linalg.inv(np.matmul(X.T,X)),my_xvector));
    tau = norm.ppf(alpha/2,loc = 0,scale = math.sqrt(temp.item(0)))
    # print('tau',tau)
    # print('confidence interval:',(y_pred+tau),(y_pred-tau))
    #2.7 d and e significant feature height and weight
    cov_theta = var_hat.item(0)*np.linalg.inv(np.matmul(X.T,X));
    # d = 3.841 # from chisquare tables
    cutoff = chi2.ppf(1 - alpha, df=1)

    test_stat_out = np.zeros(len(theta_hat));
    for i in range(1,len(theta_hat)):
        test_stat_out[i] = pow(theta_hat[i]/math.sqrt(cov_theta[i,i]),2)
        if test_stat_out[i] > cutoff:
            is_sig = 'Significant';
        else:
            is_sig = 'Not Significant';
        print(Col_names[i],': theta :',theta_hat[i] ,'. Significance:',test_stat_out[i],'. This feature is ',is_sig)
    return var_hat


#%%
def sk_linear_regression(my_xvector,df):
    print('Using Scikit linearmodel regression')
    y = df['SB_Dropout'].values
    X = df[['SB_ER','Sec_Only_Comp','SB_Toilet','Sec_Only_Elec','Sec_Only_Water']].values
    reg = linear_model.LinearRegression();
    reg.fit(X,y)
    y_pred = reg.predict(my_xvector)

    print('Coefficients: \n', reg.coef_)
    print('Intercept: \n', reg.intercept_)
    

    print('Prediction: \n', y_pred)
    return y_pred
#%% data and histograms
filename = 'school_dataset.csv'
df= read_data(filename)
df.hist(alpha=0.8, figsize=(20, 10))
corrMatrix = df.corr()
# sn.heatmap(corrMatrix, annot=True)
# plt.show()
alpha = 0.05;

# ['SB_ER','Sec_Only_Comp','SB_Toilet','Sec_Only_Elec','Sec_Only_Water']
my_xvector = np.array([100,20,90,90,90]).reshape(1, -1);

#%% Predict dropout rate using Scikits Linear regression
# All values in percentage/ratio
y_out = sk_linear_regression(my_xvector,df)
# %% PRedict with Linear regression
y_out = linear_regression(my_xvector,df)
#%% Significant features
var_hat = LR_significance(my_xvector,df,alpha)


# %%
