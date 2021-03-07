import cv2
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dlib
import imutils
from imutils import face_utils
import numpy
from openpyxl import load_workbook

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import time


# Logistic Regression
class LR:
    def __init__(self):
        self.clf = None

    def train(self, X_train, y_train):
        # clf = LogisticRegression(penalty='l2', dual='False', tol=0.0001, C=1.0,
        #     fit_intercept='True', intercept_scaling=1, class_weight='None', random_state='None',
        #     solver='warn', max_iter=100, multi_class='warn', verbose=0, warm_start='False', n_jobs='None')
        clf = LogisticRegression(C=1, multi_class="auto", solver="lbfgs", max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)
        self.clf = clf

    def predict(self, X_test):
        self.y_pred = self.clf.predict(X_test)

        return self.y_pred

    def accuracy(self, y_test):
        accuracy = (metrics.accuracy_score(y_test, self.y_pred)) * 100.0
        MSE = metrics.mean_squared_error(y_test, self.y_pred)
        print('Accuracy: {:.2f}%'.format(accuracy))
        print('MSE: {:.2f}'.format(MSE))

        return accuracy


# Support Vector Machine
class SVM:
    def __init__(self):
        self.clf = None

    def train(self, X_train, y_train):
        clf = LinearSVC(C=31, loss="squared_hinge", max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)
        self.clf = clf

    def predict(self, X_test):
        self.y_pred = self.clf.predict(X_test)

        return self.y_pred

    def accuracy(self, y_test):
        accuracy = (metrics.accuracy_score(y_test, self.y_pred)) * 100.0
        print('Accuracy: {:.2f}%'.format(accuracy))

        return accuracy


# Random Forest
class RF:
    def __init__(self):
        self.clf = None

    def train(self, X_train, y_train):
        clf = RandomForestClassifier(n_estimators=236, max_leaf_nodes=16,
                                     max_depth=4, n_jobs=-1, random_state=42)
        clf.fit(X_train, y_train)
        self.clf = clf

    def predict(self, X_test):
        self.y_pred = self.clf.predict(X_test)

        return self.y_pred

    def accuracy(self, y_test):
        accuracy = (metrics.accuracy_score(y_test, self.y_pred)) * 100.0
        print('Accuracy: {:.2f}%'.format(accuracy))

        return accuracy


# k-Nearest Neighbors
class KNN:
    def __init__(self):
        self.clf = None

    def train(self, X_train, y_train):
        clf = KNeighborsClassifier(n_neighbors=8, metric='euclidean')
        clf.fit(X_train, y_train)
        self.clf = clf

    def predict(self, X_test):
        self.y_pred = self.clf.predict(X_test)

        return self.y_pred

    def accuracy(self, y_test):
        accuracy = (metrics.accuracy_score(y_test, self.y_pred)) * 100.0
        print('Accuracy: {:.2f}%'.format(accuracy))

        return accuracy


def read_csv(csv_file):
    data = []
    with open(csv_file, 'r') as f:

        # create a list of rows in the CSV file
        rows = f.readlines()
        # strip white-space and newlines
        rows = list(map(lambda x: x.strip(), rows))

        for row in rows:
            # further split each row into columns assuming delimiter is comma
            row = row.split(',')
            # append to data-frame our new row-object with columns
            data.append(row)

    return data


# 0////////////////////////////////////////////////////////////////////////////////////////////////

# Read and safe data from CSV File
csvFile = 'data3.csv'
data = read_csv(csvFile)
wrinkles = []
labels = []
count = 0

for row in data[1:]:
    # if 0 <= int(row[1]) < 10:
    #     row[1] = 0
    # elif int(row[1]) < 20:
    #     row[1] = 1
    # elif int(row[1]) < 30:
    #     row[1] = 2
    # elif int(row[1]) < 40:
    #     row[1] = 3
    # elif int(row[1]) < 50:
    #     row[1] = 4
    # elif int(row[1]) < 60:
    #     row[1] = 5
    # elif int(row[1]) < 70:
    #     row[1] = 6
    # elif int(row[1]) < 80:
    #     row[1] = 7
    # elif int(row[1]) < 90:
    #     row[1] = 8
    # elif int(row[1]) <= 100:
    #     row[1] = 9

    count += 1
    rf = [float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]),
          float(row[7]), float(row[8]), float(row[9]), float(row[10]), float(row[11]),
          float(row[12]), float(row[13]), float(row[14]), float(row[15]), float(row[16]),
          float(row[17]), float(row[18]), float(row[19]), float(row[20]), float(row[21]),
          float(row[22]), float(row[23]), float(row[24]), float(row[25]), float(row[26]),
          float(row[27]), float(row[28]), float(row[29]), float(row[30]), float(row[31])]
    wrinkles.append(rf)
    labels.append(int(row[1]))

X_train, X_test, y_train, y_test = train_test_split(wrinkles, labels, test_size=0.2, random_state=42)

model0 = SVM()
model0.train(X_train, y_train)
model0.predict(X_test)
model0.accuracy(y_test)


# 1////////////////////////////////////////////////////////////////////////////////////////////////

# Read and safe data from CSV File
# csvFile = 'data2.csv'
# data = read_csv(csvFile)
# wrinkles = []
# labels = []
# count = 0
#
# for row in data[1:]:
#     count += 1
#     rf = [float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]),
#           float(row[7]), float(row[8]), float(row[9]), float(row[10]), float(row[11]),
#           float(row[12]), float(row[13]), float(row[14]), float(row[15]), float(row[16])]
#     wrinkles.append(rf)
#     labels.append(int(row[1]))
#
# X_train, X_test, y_train, y_test = train_test_split(wrinkles, labels, test_size=0.2, random_state=42)
#
# model1 = SVM()
# model1.train(X_train, y_train)
# model1.predict(X_test)
# model1.accuracy(y_test)


# GridSearchCV

# start_time = time.time()
# grid_algor = LinearSVC()
# param_grid = {
#     'C': list(range(20, 40, 1)),
#     'loss': ['squared_hinge'],
#     'random_state': [42],
#     'max_iter': [1000]
# }
# grid_search = GridSearchCV(grid_algor, param_grid, cv=10)
# grid_search.fit(X_train, y_train)
#
# print('Tuned hyperparameters:', grid_search.best_params_)
# print('Accuracy:', grid_search.best_score_)
# print("--- %s seconds ---" % (time.time() - start_time))