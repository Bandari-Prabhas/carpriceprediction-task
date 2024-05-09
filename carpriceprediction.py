import numpy as np
import pandas as pd

car_dataset=pd.read_csv("/content/car data.csv.xls")
car_dataset

car_dataset.info()

car_dataset.describe()

car_dataset['Fuel_Type'].value_counts()

car_dataset.isnull().sum()

car_dataset['Transmission'].value_counts()

car_dataset['Seller_Type'].value_counts()

#Encoding the Fuel_type column
car_dataset.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}},inplace=True)
#Encoding the Fuel_type column
car_dataset.replace({'Seller_Type':{'Dealer':0,'Individual':1}},inplace=True)
#Encoding the Fuel_type column
car_dataset.replace({'Transmission':{'Manual':0,'Automatic':1}},inplace=True)

car_dataset

X=car_dataset.drop(['Car_Name','Selling_Price'],axis=1)
Y=car_dataset['Selling_Price']

print(X)
print(Y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=2)
print(X.shape,X_train.shape,X_test.shape)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()

regressor.fit(X_train,Y_train)

traing_data_prediction=regressor.predict(X_train)


from sklearn import metrics
r2_train=metrics.r2_score(Y_train,traing_data_prediction)
print("R Squared Value:",r2_train)

testing_data_prediction=regressor.predict(X_test)
r2_test=metrics.r2_score(Y_test,testing_data_prediction)
print("R Squared Value:",r2_test)

input_data=(2010,0.95,27000,0,1,0,0)
input_data_as_numpy_array=np.asarray(input_data)
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

#Predicting the cost
prediction=regressor.predict(input_data_reshaped)
print("The cost of your Model car is:",prediction)