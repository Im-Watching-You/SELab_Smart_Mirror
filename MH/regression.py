import time

from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import pickle
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV


data = pd.read_csv("./data,0724.csv")
data = data.iloc[:,1:]
print(data.corr())
print(data.corr()["Age"].sort_values(ascending=False))
# up_20 = data["Age"] > 10
# down_50 = data["Age"] <= 50
# data = data[up_20 & down_50]
# data = data.iloc[:, [0, 2, 3, 4, 5, 10, 11, 12, 13, 14, 21, 24, 25, 28]]
# print(data.iloc[:,-1])
# data.loc[:, 'Age'] = data.loc[:, "Age"] // 10
# print(data.describe())
# print(data.corr()["Age"].sort_values(ascending=False))

train, test = train_test_split(data)
X, y = data.iloc[:, [2,3,4,5, 6,7,8,9,10,11,12]], data.loc[:, "Age"]
X_train, y_train = train.iloc[:, [2,3,4,5, 6,7,8,9,10,11,12]], train.loc[:, "Age"]
X_test, y_test = test.iloc[:, [2,3,4,5,6,7,8,9,10,11,12]], test.loc[:, "Age"]
##############################################################################################
start_time = time.time()
clf = SVR(gamma='scale')
params = {'C': [1.0, 2.0, 3.0, 5.0, 10.0, 50.0, 100.0, 500], 'epsilon': [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0],
          'kernel': ['linear', 'poly', 'rbf']}
grc = GridSearchCV(clf, params, cv=5)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# joblib.dump(clf, "SVC.pkl")
grc.fit(X, y)
cvres = grc.cv_results_
for acc, params in zip(cvres['mean_test_score'], cvres['params']):
    print(acc, params)
print(grc.best_estimator_, grc.best_score_, grc.best_params_, )
print("Elapsed Time: ", round(time.time()-start_time, 2))
print("===========================================================")
start_time = time.time()
neigh = KNeighborsRegressor()
params = {'n_neighbors': [3, 5, 7, 13, 21, 51, 101, 161]}
grc = GridSearchCV(neigh, params, cv=5)
grc.fit(X, y)
cvres = grc.cv_results_
for acc, params in zip(cvres['mean_test_score'], cvres['params']):
    print(acc, params)
print(grc.best_estimator_, grc.best_score_, grc.best_params_, )
print("Elapsed Time: ", round(time.time()-start_time, 2))
print("===========================================================")
start_time = time.time()

# print(accuracy_score(y_test, y_pred))
# print(mean_squared_error(y_test, y_pred))
params = {'max_depth': [5, 10, 20, 40, 50, 100]}
dtc = DecisionTreeRegressor()
grc = GridSearchCV(dtc, params, cv=5)
grc.fit(X, y)
cvres = grc.cv_results_
for acc, params in zip(cvres['mean_test_score'], cvres['params']):
    print(acc, params)
print(grc.best_estimator_, grc.best_score_, grc.best_params_, )
# dtc.fit(X_train, y_train)
# y_pred = dtc.predict(X_test)

print("Elapsed Time: ", round(time.time()-start_time, 2))
print("===========================================================")
# joblib.dump(dtc, "dtc.pkl")
# print(accuracy_score(y_test, y_pred))
# print(mean_squared_error(y_test, y_pred))

params = {'C': [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]}
dtc = LinearRegression()
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(rmse)
# # lin_reg = LinearRegression()
# # lin_reg.fit(X_train, y_train)
#
# joblib.dump(dtc, "lr.pkl")
# print(accuracy_score(y_test, y_pred))
# print(mean_squared_error(y_test, y_pred))
#
# y_pred = lin_reg.predict(X_test)
# f = np.array([147, 73, 37, 20, 106, 48, 78, 10, 42, 85, 50, 139])
# f = np.reshape(f, (1, -1))
# # f = np.reshape(f, (-1, 1))
# age = dtc.predict(f)

# print("age: ", age)
# print(dtc.score(X_test, y_test))
# grc = GridSearchCV(dtc, params, cv=5, scoring="accuracy")
# grc.fit(X, y)
# cvres = grc.cv_results_
# for acc, params in zip(cvres['mean_test_score'], cvres['params']):
#     print(acc, params)
# print(grc.best_estimator_, grc.best_score_, grc.best_params_, )
# dtc.fit(X_train, y_train)
# y_pred = dtc.predict(X_test)

print("Elapsed Time: ", round(time.time()-start_time, 2))
print("===========================================================")