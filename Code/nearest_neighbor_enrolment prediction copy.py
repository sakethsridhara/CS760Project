
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, metrics
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import  cross_val_score
from sklearn.model_selection import train_test_split
from scipy.stats import norm
from scipy.stats.distributions import chi2
from sklearn.neighbors import KNeighborsRegressor

import math

#%%
def read_data(filename):
    df = pd.read_csv(filename)
    df['Bias'] = 1;
    return df
#%%
def read_data_with_fields(filename,fields):
    df = pd.read_csv(filename,usecols = fields)
    df['Bias'] = 1;
    return df
#%%
def linear_regression(my_xvector,df,alpha,fields):
    # print('Linear regression using own code:')
    y = df[fields[0]].values
    # fields.insert(1,'Bias')
    Col_names = fields[1:9]   
    X = df[Col_names].values
    theta_hat = np.matmul(np.linalg.inv(np.matmul(X.T,X)),np.matmul(X.T,y));
    print(theta_hat)
    N = len(y);
    my_xvector = np.insert(my_xvector, [0], 1);
    y_pred = np.matmul(my_xvector.T, theta_hat);
    # print('predicted enrollment rate:',y_pred)
    var_hat = (1.0/N)*np.matmul((y - np.matmul(X,theta_hat)).T,(y - np.matmul(X,theta_hat)));
    # print('var_hat',var_hat)
    temp = var_hat*np.matmul(my_xvector.T,np.matmul(np.linalg.inv(np.matmul(X.T,X)),my_xvector));
    tau = norm.ppf(alpha/2,loc = 0,scale = math.sqrt(temp.item(0)))
    # print('tau',tau)
    # rint('confidence interval:',(y_pred+tau),(y_pred-tau))
    y_lb = y_pred+tau
    y_ub = y_pred-tau
    return y_pred,y_lb, y_ub
#%%
def LR_significance(my_xvector,df,alpha,fields):
    y = df[fields[0]].values
    # fields.insert(1,'Bias')
    Col_names = fields[1:9]   
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
def sk_linear_regression(my_xvector,df,fields):
    print('Using Scikit linearmodel regression')
    y = df[fields[0]].values
    X = df[fields[2:9]].values
    reg = linear_model.LinearRegression();
    reg.fit(X,y)
    y_pred = reg.predict(my_xvector)

    print('Coefficients: \n', reg.coef_)
    print('Intercept: \n', reg.intercept_)
    

    print('Prediction: \n', y_pred)
    return y_pred


#%% 
def LR_CV(df,n_folds,alpha,fields):
    fold_size = math.floor(len(df)/n_folds)
    list_of_dfs = [df.loc[i:i+fold_size-1,:] for i in range(0, len(df),fold_size)]
    error_count = 0;
    for i in range(n_folds):
        test_df = list_of_dfs[i]
        train_df = df.drop(df.index[test_df.index[0]:test_df.index[-1]])
        j = 0;
        for j in range(len(test_df)):
            truth = test_df.values[j][0];
            prediction, pred_lb, pred_ub = linear_regression((test_df.values[j][1:8]), train_df, alpha, fields);
            if(pred_lb <= truth <= pred_ub):
                # print('Me here')
                error_count += 0;
            else:
                # print('Me here')
                error_count += 1;
    CV_out = 100*(1-(error_count/len(df)));
    print('Accuracy with cross-validation =',CV_out);
    return CV_out

#%%
def LR_CV2(df, fields, n_folds):
    y = df[fields[0]].values
    X = df[fields[1:8]].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    # reg = linear_model.LinearRegression() 
    # reg.fit(X_train, y_train)
    reg = KNeighborsRegressor(n_neighbors=2)
    reg.fit(X_train, y_train)
    scores = cross_val_score(reg, X_train, y_train, cv = n_folds)
    print("mean cross validation score: {}".format(np.mean(scores)))
    print("score without cv: {}".format(reg.score(X_train, y_train)))
    
    y_pred = reg.predict(X_test)
    print('R2 score:',r2_score(y_test, y_pred))
    print('regressor score:',reg.score(X_test, y_test))
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))) 

#%% data and histograms
filename = 'Reduced_data_district_2.csv'
fields =['P_Enrollment_Rate','Percent_schools_with_boys_toilet','Percent_schools_with_playground',\
        'Percent_schools_with_drinking_water','Percent_schools_with_electricity',\
            'Percent_schools_with_Roads','Percent_classrooms_requiring_major repair','Percent_single_teacher_schools']
df= read_data_with_fields(filename,fields)
df = df[df.P_Enrollment_Rate.notnull()]
df.P_Enrollment_Rate = pd.to_numeric(df.P_Enrollment_Rate, errors='coerce')
df = df.dropna()
# fields.insert(1,'Bias')
df.hist(alpha=0.8)
my_xvector = np.array([65,60,96,57,87,9,7]).reshape(1, -1);
alpha = 0.05;
n_folds = 5;

#%% Predict dropout rate using Scikits Linear regression
y_out,y_lb,y_ub = linear_regression(my_xvector,df,alpha,fields)
#%% using SKlearn
# All values in percentage/ratio
y_out = sk_linear_regression(my_xvector,df,fields)
# %% PRedict with Linear regression
# y_out = linear_regression(my_xvector,df)
#%% Significant features
var_hat = LR_significance(my_xvector,df,alpha,fields)


# %% Cross validation of linear regression
CV_out = LR_CV(df,n_folds,alpha,fields);


# %%
LR_CV2(df, fields, n_folds)
# %%
