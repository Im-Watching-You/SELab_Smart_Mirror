"""
Date: 2019.07.24
Programmer: Maksym Chernozhukov
Description: Code for Aging Analytics "Prediction Age by Wrinkle and Aging Spot Features"
"""

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
import time

import cv2
import math
import numpy as np
import pandas as pd
import dlib
import pickle
import imutils
import os.path
from imutils import face_utils
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

class AgingDiagnosis:
    def __init__(self):
        self.wrinkles = []
        self.labels = []

    def read_csv(self, csv_file):
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

    def save_model(self, pickle_path):
        if os.path.isfile(pickle_path) == False:
            file_wr = open(pickle_path, 'wb')
            pickle.dump(self.clf, file_wr)
            file_wr.close()

    def read_model(self, pickle_path):
        file_r = open(pickle_path, 'rb')
        self.trained_model = pickle.load(file_r)
        return self.trained_model

    def make_preprocessing(self, csvFile):
        data = self.read_csv(csvFile)
        # Rescaling Size and Count of Aging Spots
        cd = 100.0

        for row in data[1:]:
            rf = [float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]),
                  float(row[7]), float(row[8]), float(row[9]), float(row[10]), float(row[11]),
                  float(row[12]), float(row[13]), float(row[14]), float(row[15]), float(row[16]),
                  float(row[17]), float(row[18]), float(row[19]), float(row[20]), float(row[21]),
                  float(row[22]), float(row[23]), float(row[24]), float(row[25]), float(row[26]),
                  float(row[27])/cd, float(row[28])/cd, float(row[29])/cd, float(row[30])/cd, float(row[31])/cd,
                  float(row[32]), float(row[33]), float(row[34]), float(row[35]), float(row[36]), float(row[37]),
                  float(row[38]), float(row[39]), float(row[40]), float(row[41]), float(row[42]), float(row[43])]

            self.wrinkles.append(rf)
            self.labels.append(int(row[1]))

    def seperate_dataset(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.wrinkles, self.labels,
                                                                                test_size=0.2, random_state=42)

        return self.X_train, self.X_test, self.y_train, self.y_test

    # //////////////////////////////////////////////////////////////////////////////

    def train_model(self):
        self.clf = RandomForestClassifier()
        self.clf.fit(self.X_train, self.y_train)

        return self.clf

    # //////////////////////////////////////////////////////////////////////////////

    def estimate_model(self):
        y_pred = self.clf.predict(self.X_test)
        accuracy = (metrics.accuracy_score(self.y_test, y_pred)) * 100.0
        print('Accuracy: {:.2f}%'.format(accuracy))


if __name__ == '__main__':
    csvFile = './data/data7.csv'
    model_path = './trained_models/trained_model7.pkl'

    # Make trained model, estimate and save the model in a file
    aging_diag = AgingDiagnosis()
    aging_diag.make_preprocessing(csvFile)
    X_train, X_test, y_train, y_test = aging_diag.seperate_dataset()

    aging_diag.train_model()
    aging_diag.estimate_model()
    # aging_diag.save_model(model_path)
    # aging_diag.read_model(model_path)

# Accuracy: 3.80%
# Accuracy: 3.71%

# # # GridSearchCV
#     start_time = time.time()
#     grid_algor = RandomForestClassifier()
#     param_grid = {
#             'n_estimators': list(range(1, 100, 1))
#     }
#     grid_search = GridSearchCV(grid_algor, param_grid, cv=10)
#     grid_search.fit(X_train, y_train)
#
#     print('Tuned hyperparameters:', grid_search.best_params_)
#     print('Accuracy:', grid_search.best_score_)
#     print("--- %s seconds ---" % (time.time() - start_time))




# # GridSearchCV
    # start_time = time.time()
    # grid_algor = RandomForestClassifier()
    # param_grid = {
    #         'n_estimators': list(range(1, 500, 10)),
    #         'max_leaf_nodes': [2, 4, 6, 8, 12, 16],
    #         'max_depth': [2, 4, 8],
    #         'random_state': [42]
    # }
    # grid_search = GridSearchCV(grid_algor, param_grid, cv=10)
    # grid_search.fit(X_train, y_train)
    #
    # print('Tuned hyperparameters:', grid_search.best_params_)
    # print('Accuracy:', grid_search.best_score_)
    # print("--- %s seconds ---" % (time.time() - start_time))


    # 'C': list(range(20, 40, 1)),
    # 'loss': ['squared_hinge'],
    # 'random_state': [42],
    # 'max_iter': [1000]
    # LR


    # # Correlation
    # data = pd.read_csv("./data/new_data.csv")
    # data = data.iloc[:, 1:]
    # corr = data.corr()
    # print(corr['Age'].sort_values(ascending=False))

# self.clf = LogisticRegression(C=1, multi_class="auto", solver="lbfgs", max_iter=1000, random_state=42)

# self.clf = LinearSVC(C=31, loss="squared_hinge", max_iter=1000, random_state=42)

# self.clf = RandomForestClassifier(n_estimators=236, max_leaf_nodes=16,
#                                      max_depth=4, n_jobs=-1, random_state=42)

# self.clf = KNeighborsClassifier(n_neighbors=8, metric='euclidean')


    # Accuracy: 3.50 %
    # SVM
    # Accuracy: 1.90 %
    # RF
    # Accuracy: 3.54 %
    # KNN
    # Accuracy: 2.89 %

    # def display_scores(scores, clf):
    #     print("\n" + clf.__class__.__name__ + ":")
    #     print(" Scores:", scores)
    #     print("Mean:", scores.mean())
    #     print("Standard deviation:", scores.std())
    #
    # log_clf = LogisticRegression()
    # svm_clf = LinearSVC()
    # rnd_clf = RandomForestClassifier(random_state=42)
    # knn_clf = KNeighborsClassifier()
    #
    # voting_clf = VotingClassifier(
    #     estimators=[('lr', log_clf), ('svm', svm_clf), ('rf', rnd_clf), ('knn', knn_clf)],
    #     voting='hard')
    # voting_clf.fit(X_train, y_train)
    #
    # for clf in (log_clf, svm_clf, rnd_clf, knn_clf, voting_clf):
    #     svm_scores = cross_val_score(clf, X_train, y_train, cv=10, scoring="accuracy")
    #     display_scores(svm_scores, clf)
    #
    #     clf.fit(X_train, y_train)
    #     y_pred = clf.predict(X_test)
    #     print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

# LogisticRegression 0.0367170626349892
# LinearSVC 0.031101511879049675
# RandomForestClassifier 0.03758099352051836
# KNeighborsClassifier 0.02678185745140389

# VotingClassifier 0.031965442764578834
# VotingClassifier 0.033693304535637146  With out KNN
# VotingClassifier 0.037149028077753776  Wirh out SVM and KNN


# LogisticRegression:
#  Scores: [0.02368692 0.02919708 0.02944269 0.03609342 0.03675676 0.02508179
#  0.03736264 0.02774695 0.03579418 0.04288939]
# Mean: 0.032405181551463956
# Standard deviation: 0.005907005969993041
# LogisticRegression 0.0367170626349892

# LinearSVC:
#  Scores: [0.02986612 0.03858186 0.03470032 0.03609342 0.03351351 0.03271538
#  0.02417582 0.02330744 0.03467562 0.02708804]
# Mean: 0.0314717508649372
# Standard deviation: 0.004898966983836273


# RandomForestClassifier:
#  Scores: [0.03295572 0.03441084 0.02839117 0.0329087  0.03135135 0.03598691
#  0.03516484 0.03773585 0.03355705 0.04514673]
# Mean: 0.034760915572689836
# Standard deviation: 0.004228967141783403

# RandomForestClassifier:
#  Scores: [0.03501545 0.03336809 0.03364879 0.0477707  0.03351351 0.03816794
#  0.04285714 0.03662597 0.03467562 0.03837472]
# Mean: 0.03740179306281521
# Standard deviation: 0.004449850854818415
# RandomForestClassifier 0.03628509719222462



# KNeighborsClassifier:
#  Scores: [0.02471679 0.02711157 0.03890641 0.0329087  0.03027027 0.03053435
#  0.03186813 0.03662597 0.04250559 0.03837472]
# Mean: 0.03338225156592419
# Standard deviation: 0.005328429457495356

# VotingClassifier:
#  Scores: [0.02677652 0.02815433 0.03575184 0.03609342 0.02918919 0.0348964
#  0.03186813 0.02663707 0.03803132 0.03950339]
# Mean: 0.03269016031074648
# Standard deviation: 0.004537498688553791




    # def display_scores(scores, clf):
    #         print("\n" + clf.__class__.__name__ + ":")
    #         print(" Scores:", scores)
    #         print("Mean:", scores.mean())
    #         print("Standard deviation:", scores.std())
    #
    # rnd_clf = RandomForestClassifier()
    #
    # svm_scores = cross_val_score(rnd_clf, X_train, y_train, cv=10, scoring="accuracy")
    # display_scores(svm_scores, rnd_clf)
    #
    # rnd_clf.fit(X_train, y_train)
    # y_pred = rnd_clf.predict(X_test)
    # print(rnd_clf.__class__.__name__, accuracy_score(y_test, y_pred))