
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
%matplotlib inline 

USAhouse = pd.read_csv('C:/Users/lucas/Downloads/archive (1)/USA_Housing.csv')

USAhouse.head()

USAhouse.info()

X = USAhouse[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms','Avg. Area Number of Bedrooms', 'Area Population']]
Y = USAhouse['Price']       

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.4,random_state=101)

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train,Y_train)

print(lm.intercept_)

print(lm.coef_)

coefs = pd.DataFrame(lm.coef_, X.columns, columns = ['Coefs'])

coefs

predict = lm.predict(X_test)


from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(Y_test,predict))
print('MSE:', metrics.mean_squared_error(Y_test,predict))
print('RMSE:', np.sqrt(metrics.mean_squared_error(Y_test,predict)))

