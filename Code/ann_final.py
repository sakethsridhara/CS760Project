# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import the regressor 
from sklearn import metrics
from sklearn.model_selection import  cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from keras.models import Sequential
import keras
from keras.layers import Dense
from sklearn.model_selection import KFold
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler
from numpy import array
from sklearn.preprocessing import  MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

def read_data_with_fields(filename,fields):
    df = pd.read_csv(filename,usecols = fields)
    return df

filename = 'Reduced_data_district_2.csv'
fields =['FEMALE_LIT','Percent_schools_with_boys_toilet','Percent_schools_with_playground',\
        'Percent_schools_with_drinking_water','Percent_schools_with_electricity','Percent_schools_with_Roads', 'Percent_single_teacher_schools']
df= read_data_with_fields(filename,fields)

df = df.dropna()
y = df[fields[0]].values
X = df[fields[1:7]].values


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 1)

sc= MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

model = Sequential()
model.add(Dense(10, input_dim=6, activation='relu'))
model.add(Dense(20,  activation='relu'))
model.add(Dense(10,  activation='relu'))
model.add(Dense(5,  activation='relu'))
model.add(Dense(1, activation='linear'))

keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='mean_squared_error', optimizer='RMSprop', metrics=['mean_absolute_percentage_error', 'mse'])

history = model.fit(X_train, y_train, epochs=2000, batch_size=32,validation_split=0.15,validation_data=None,verbose=1)

plt.ylim(60, 110)
plt.plot(model.predict(X_test))
plt.plot(y_test)
plt.ylim(60, 110)
plt.plot(model.predict(X_train))
plt.plot(y_train)

y_pred = np.concatenate(model.predict(X_test)).ravel().tolist()

y_true = y_test.ravel().tolist()


print(PCC = pearsonr(y_pred, y_true))

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

my_xvector =  np.array([65,60,96,57,87,7]).reshape(1, -1);
my_xvector = sc.fit_transform(my_xvector)

y_out = model.predict(my_xvector)


