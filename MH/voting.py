import time

from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import pickle
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier, GradientBoostingClassifier


def compare_prediction(val_true, val_pred):
    if type(val_true) != list:
        val_true = val_true.tolist()
    diff = []
    mse = mean_squared_error(val_true, val_pred)
    rmse = np.sqrt(mse)
    for i in range(len(val_true)):
        diff.append(abs(val_true[i]-val_pred[i]))

    mean = np.mean(diff)
    std = np.std(diff)
    var = np.var(diff)

    for i in range(len(val_pred)):
        val_pred[i] = int(val_pred[i])

    acc = accuracy_score(val_true, val_pred)

    result = {"rmse":rmse, "max": max(diff), 'min': min(diff), "mean": mean, "std":std, "var": var, "age_accuracy": round(1-(mean/100),2), "accuracy": acc}
    return result



files = ["./data,0724.csv", "./data, FGN, 0730.csv", "./data, FGN, 0730b.csv", "./data, FGN, 0730c.csv",
         "./data_new, 0730.csv", "./data_new, 0730c.csv"]
for f in files:
    print("Current Model Name: ", f)
    data = pd.read_csv(f)
    data = data.iloc[:, 1:]
    # print(data.corr())
    print(data.corr()["Age"].sort_values(ascending=False))
    # print(data.head())
    # up_20 = data["Age"] > 10
    # down_50 = data["Age"] <= 50
    # data = data[up_20 & down_50]
    # data = data.iloc[:, [0, 2, 3, 4, 5, 10, 11, 12, 13, 14, 21, 24, 25, 28]]
    # print(data.iloc[:,-1])
    # data.loc[:, 'Age'] = data.loc[:, "Age"] // 10
    # print(data.describe())
    # print(data.corr()["Age"].sort_values(ascending=False))
    f_name = f.split(".")[0]
    train, test = train_test_split(data)
    X, y = data.iloc[:, [1,2,3, 4, 5,6,7, 8, 9,10,11, 12]], data.loc[:, "Age"]
    X_train, y_train = train.iloc[:, [1,2,3, 4, 5,6,7, 8, 9,10,11, 12]], train.loc[:, "Age"]
    X_test, y_test = test.iloc[:, [1,2,3, 4, 5,6,7, 8, 9,10,11, 12]], test.loc[:, "Age"]

    rdf = RandomForestClassifier(n_estimators=10, random_state=1)
    clf = SVC(gamma='scale', C=50, kernel="rbf", probability=True)
    neigh = KNeighborsClassifier(n_neighbors=21)
    dct = DecisionTreeClassifier(max_depth=10)
    dtc = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', C=50)
    print("====================================================================\n")
    start_time = time.time()
    print("RandomForestClassifier")
    rdf.fit(X_train, y_train)
    y_pred = rdf.predict(X_test)
    joblib.dump(rdf, "RDF"+f_name+".pkl")
    print(compare_prediction(y_test, y_pred))
    print("Elapsed Time: ", round(time.time()-start_time, 2))
    print("====================================================================\n")
    start_time = time.time()
    print("DecisionTreeClassifier")
    dct.fit(X_train, y_train)
    y_pred = dct.predict(X_test)
    joblib.dump(dct, "DCT"+f_name+".pkl")
    print(compare_prediction(y_test, y_pred))
    print("Elapsed Time: ", round(time.time()-start_time, 2))
    print("====================================================================\n")
    start_time = time.time()
    print("KNN")
    neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_test)
    joblib.dump(neigh, "KNN"+f_name+".pkl")
    print(compare_prediction(y_test, y_pred))
    print("Elapsed Time: ", round(time.time()-start_time, 2))
    print("====================================================================\n")
    start_time = time.time()
    print("SVC")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    joblib.dump(SVC, "SVC"+f_name+".pkl")
    print(compare_prediction(y_test, y_pred))
    print("Elapsed Time: ", round(time.time()-start_time, 2))
    print("====================================================================\n")
    start_time = time.time()
    print("LogisticRegression")
    dtc.fit(X_train, y_train)
    y_pred = dtc.predict(X_test)
    joblib.dump(dtc, "LR"+f_name+".pkl")
    print(compare_prediction(y_test, y_pred))
    print("Elapsed Time: ", round(time.time()-start_time, 2))
    print("====================================================================\n\n")
    start_time = time.time()
    print("Hard Voting")
    vt = VotingClassifier(estimators=[('rdf', rdf), ('knn', clf), ('neigh', neigh), ('dct', dct), ('lr', dtc)], voting="hard")
    vt.fit(X_train, y_train)
    y_pred = vt.predict(X_test)
    joblib.dump(vt, "VT_Hard"+f_name+".pkl")
    print(compare_prediction(y_test, y_pred))
    print("Elapsed Time: ", round(time.time()-start_time, 2))
    print("====================================================================\n")
    start_time = time.time()
    print("Soft Voting")
    vt = VotingClassifier(estimators=[('rdf', rdf), ('knn', clf), ('neigh', neigh), ('dct', dct), ('lr', dtc)], voting="soft")
    vt.fit(X_train, y_train)
    y_pred = vt.predict(X_test)
    joblib.dump(vt, "VT_soft"+f_name+".pkl")
    print(compare_prediction(y_test, y_pred))
    print("Elapsed Time: ", round(time.time()-start_time, 2))
    print("====================================================================\n")
    start_time = time.time()
    print("Ada")
    vt = AdaBoostClassifier(n_estimators=100, random_state=0)
    vt.fit(X_train, y_train)
    y_pred = vt.predict(X_test)
    joblib.dump(vt, "Ada"+f_name+".pkl")
    print(compare_prediction(y_test, y_pred))
    print("Elapsed Time: ", round(time.time()-start_time, 2))
    print("====================================================================\n")
    start_time = time.time()
    print("Gradient")
    vt = GradientBoostingClassifier()
    vt.fit(X_train, y_train)
    y_pred = vt.predict(X_test)
    joblib.dump(vt, "Gradient"+f_name+".pkl")
    print(compare_prediction(y_test, y_pred))
    print("Elapsed Time: ", round(time.time()-start_time, 2))
    print("====================================================================\n\n\n")

#
# start_time = time.time()
# clf = RandomForestClassifier(random_state=1)
# params = {'n_estimators': [3, 5, 10, 20, 50, 100]}
# grc = GridSearchCV(clf, params, cv=5, scoring="accuracy")
# # clf.fit(X_train, y_train)
# # y_pred = clf.predict(X_test)
# # joblib.dump(clf, "SVC.pkl")
# grc.fit(X, y)
# cvres = grc.cv_results_
# for acc, params in zip(cvres['mean_test_score'], cvres['params']):
#     print(acc, params)
# print(grc.best_estimator_, grc.best_score_, grc.best_params_, )
# print("Elapsed Time: ", round(time.time()-start_time, 2))
# print("===========================================================")
# start_time = time.time()
# clf = SVC(gamma='scale')
# params = {'C': [1.0, 2.0, 3.0, 5.0, 10.0, 50.0, 100.0, 500], 'kernel': ('linear', 'rbf')}
# grc = GridSearchCV(clf, params, cv=5, scoring="accuracy")
# # clf.fit(X_train, y_train)
# # y_pred = clf.predict(X_test)
# # joblib.dump(clf, "SVC.pkl")
# grc.fit(X, y)
# cvres = grc.cv_results_
# for acc, params in zip(cvres['mean_test_score'], cvres['params']):
#     print(acc, params)
# print(grc.best_estimator_, grc.best_score_, grc.best_params_, )
# print("Elapsed Time: ", round(time.time()-start_time, 2))
# print("===========================================================")
# start_time = time.time()
# neigh = KNeighborsClassifier()
# params = {'n_neighbors': [3, 5, 7, 13, 21, 51, 101]}
# grc = GridSearchCV(neigh, params, cv=5, scoring="accuracy")
# grc.fit(X, y)
# cvres = grc.cv_results_
# for acc, params in zip(cvres['mean_test_score'], cvres['params']):
#     print(acc, params)
# print(grc.best_estimator_, grc.best_score_, grc.best_params_, )
# print("Elapsed Time: ", round(time.time()-start_time, 2))
# print("===========================================================")
# start_time = time.time()
#
# # print(accuracy_score(y_test, y_pred))
# # print(mean_squared_error(y_test, y_pred))
# params = {'max_depth': [5, 10, 20, 40, 50, 100]}
# dtc = DecisionTreeClassifier()
# grc = GridSearchCV(dtc, params, cv=5, scoring="accuracy")
# grc.fit(X, y)
# cvres = grc.cv_results_
# for acc, params in zip(cvres['mean_test_score'], cvres['params']):
#     print(acc, params)
# print(grc.best_estimator_, grc.best_score_, grc.best_params_, )
# # dtc.fit(X_train, y_train)
# # y_pred = dtc.predict(X_test)
#
# print("Elapsed Time: ", round(time.time()-start_time, 2))
# print("===========================================================")
# # joblib.dump(dtc, "dtc.pkl")
# # print(accuracy_score(y_test, y_pred))
# # print(mean_squared_error(y_test, y_pred))
#
# params = {'C': [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]}
# dtc = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
# # dtc.fit(X_train, y_train)
# # y_pred = dtc.predict(X_test)
# # # lin_reg = LinearRegression()
# # # lin_reg.fit(X_train, y_train)
# #
# # joblib.dump(dtc, "lr.pkl")
# # print(accuracy_score(y_test, y_pred))
# # print(mean_squared_error(y_test, y_pred))
# #
# # y_pred = lin_reg.predict(X_test)
# # f = np.array([147, 73, 37, 20, 106, 48, 78, 10, 42, 85, 50, 139])
# # f = np.reshape(f, (1, -1))
# # # f = np.reshape(f, (-1, 1))
# # age = dtc.predict(f)
#
# # print("age: ", age)
# # print(dtc.score(X_test, y_test))
# grc = GridSearchCV(dtc, params, cv=5, scoring="accuracy")
# grc.fit(X, y)
# cvres = grc.cv_results_
# for acc, params in zip(cvres['mean_test_score'], cvres['params']):
#     print(acc, params)
# print(grc.best_estimator_, grc.best_score_, grc.best_params_, )
# # dtc.fit(X_train, y_train)
# # y_pred = dtc.predict(X_test)
#
# print("Elapsed Time: ", round(time.time()-start_time, 2))
# print("===========================================================")
# params = {'max_depth': [5, 10, 20, 40, 50, 100]}
# dtc = DecisionTreeClassifier()
# grc = GridSearchCV(dtc, params, cv=5, scoring="accuracy")
# grc.fit(X, y)
# cvres = grc.cv_results_
# for acc, params in zip(cvres['mean_test_score'], cvres['params']):
#     print(acc, params)
# print(grc.best_estimator_, grc.best_score_, grc.best_params_, )
# # dtc.fit(X_train, y_train)
# # y_pred = dtc.predict(X_test)
#
# print("Elapsed Time: ", round(time.time()-start_time, 2))
# print("===========================================================")