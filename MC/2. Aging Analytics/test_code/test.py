"""
Date: 2019.07.17
Programmer: Maksym Chernozhukov
Description: Code for Aging Analytics "Extracting Wrinkle Features"
"""

import cv2
import os
import math
import numpy as np
import pandas as pd
import time
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
import pickle


"""
    1. Functions
"""


# Sobel Filter for Detection Wrinkles
def apply_sobel(img, ksize, thres=None):
    Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
    Iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
    Im = cv2.magnitude(Ix, Iy)
    if thres is not None:
        _, It = cv2.threshold(Im, thres, 1, cv2.THRESH_BINARY)
        return It
    else:
        return Im


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def wrinkle_density(img, threshold):
    Wa = np.sum(img >= threshold)
    Pa = img.shape[0] * img.shape[1]
    result = Wa/Pa
    return result


def wrinkle_depth(img, threshold):
    Wa = img[img >= threshold]
    M = np.sum(Wa)
    result = M / (255*len(Wa))
    return result


def avg_skin_variance(img):
    M = np.sum(img)
    Pa = img.shape[0] * img.shape[1]
    result = M / (255*Pa)
    return result

# Read and safe data from CSV File


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


def save_model(pickle_path, trained_model):
    file_wr = open(pickle_path, 'wb')
    pickle.dump(trained_model, file_wr)
    file_wr.close()


def open_model(pickle_path):
    file_r = open(pickle_path, 'rb')
    trained_model = pickle.load(file_r)
    return trained_model


"""
    2. Extracting Features
"""


def extract_frinkles(face_image):

    predictor_path = 'shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    threshold = 40

    try:
        image = cv2.imread(face_image)
        image = imutils.resize(image, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)

        shape = predictor(gray, rects[0])
        shape = face_utils.shape_to_np(shape)

        # Find Wrinle Sections
        corner_left_eye = [shape[2][0], min(shape[37][1], shape[38][1]), shape[18][0], shape[28][1]]
        corner_right_eye = [shape[25][0], min(shape[43][1], shape[44][1]), shape[14][0], shape[28][1]]
        forehead = [shape[19][0], shape[19][1] - (shape[42][0] - shape[39][0]),
                    shape[24][0], min(shape[19][1], shape[24][1])]
        cheek_left = [shape[17][0], shape[28][1], shape[49][0], shape[2][1]]
        cheek_right = [shape[53][0], shape[28][1], shape[26][0], shape[14][1]]

        # Apply Sobel Filter to get wrinkled image
        wrinkled = apply_sobel(gray, 3)

        # Compute Wrinkles
        window = wrinkled[corner_left_eye[1]:corner_left_eye[3], corner_left_eye[0]:corner_left_eye[2]]
        left_eye_wr = wrinkle_density(window, threshold)
        d1 = wrinkle_depth(window, threshold)
        v1 = avg_skin_variance(window)

        window = wrinkled[corner_right_eye[1]:corner_right_eye[3], corner_right_eye[0]:corner_right_eye[2]]
        right_eye_wr = wrinkle_density(window, threshold)
        d2 = wrinkle_depth(window, threshold)
        v2 = avg_skin_variance(window)

        window = wrinkled[forehead[1]:forehead[3], forehead[0]:forehead[2]]
        forehead_wr = wrinkle_density(window, threshold)
        d3 = wrinkle_depth(window, threshold)
        v3 = avg_skin_variance(window)

        window = wrinkled[cheek_left[1]:cheek_left[3], cheek_left[0]:cheek_left[2]]
        cheek_left_wr = wrinkle_density(window, threshold)
        d4 = wrinkle_depth(window, threshold)
        v4 = avg_skin_variance(window)

        window = wrinkled[cheek_right[1]:cheek_right[3], cheek_right[0]:cheek_right[2]]
        cheek_right_wr = wrinkle_density(window, threshold)
        d5 = wrinkle_depth(window, threshold)
        v5 = avg_skin_variance(window)

        # Skip NaN
        if math.isnan(left_eye_wr) or math.isnan(right_eye_wr) or math.isnan(forehead_wr) or\
                math.isnan(cheek_left_wr) or math.isnan(cheek_right_wr):
            print('Take another picture')

        if math.isnan(d1) or math.isnan(d2) or math.isnan(d3) or math.isnan(d4) or math.isnan(d5):
            print('Take another picture')

        if math.isnan(v1) or math.isnan(v2) or math.isnan(v3) or math.isnan(v4) or math.isnan(v5):
            print('Take another picture')

        # Corner Left Eye, Corner Right Eye, Forehead Values > 0.8 generally Error (hair, sunglasses, and etc.)
        if left_eye_wr > 0.8:
            print('May be ERROR')
            left_eye_wr = (cheek_left_wr + cheek_right_wr) * 0.5
        if right_eye_wr > 0.8:
            print('May be ERROR')
            right_eye_wr = (cheek_left_wr + cheek_right_wr) * 0.5
        if forehead_wr > 0.8:
            print('May be ERROR')
            forehead_wr = (cheek_left_wr + cheek_right_wr) * 0.5

        result = [[left_eye_wr, right_eye_wr, forehead_wr, cheek_left_wr, cheek_right_wr,
                  d1, d2, d3, d4, d5,
                  v1, v2, v3, v4, v5]]

        return result
    except:
        print('Try again')


class AgingDiagnosis:
    def __init__(self):
        self.wrinkles = []
        self.labels = []

    def make_preprocessing(self, csvFile):
        data = read_csv(csvFile)

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
            rf = [float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]),
                  float(row[7]), float(row[8]), float(row[9]), float(row[10]), float(row[11]),
                  float(row[12]), float(row[13]), float(row[14]), float(row[15]), float(row[16])]
            self.wrinkles.append(rf)
            self.labels.append(int(row[1]))

    def seperate_dataset(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.wrinkles, self.labels,
                                                                                test_size=0.2, random_state=42)

    def train_model(self):
        self.clf = LinearSVC(C=31, loss="squared_hinge", max_iter=1000, random_state=42)
        self.clf.fit(self.X_train, self.y_train)
        return self.clf

    def estimate_model(self):
        y_pred = self.clf.predict(self.X_test)
        accuracy = (metrics.accuracy_score(self.y_test, y_pred)) * 100.0
        print('Accuracy: {:.2f}%'.format(accuracy))

    def analyse_factors(self, trained_model, face_image):
        features = extract_frinkles(face_image)
        y_pred = trained_model.predict(features)
        result = {'Age': y_pred}

        return result


csvFile = 'data.csv'
face_image = './datasets/experement/max1.jpg'
pickle_path = './trained_model.pkl'

model = AgingDiagnosis()
# model.make_preprocessing(csvFile)
# model.seperate_dataset()
#
# trained_model = model.train_model()
# model.estimate_model()
# save_model(pickle_path, trained_model)

trained_model = open_model(pickle_path)
value = model.analyse_factors(trained_model, face_image)

# Result for Case 1, and Case 5
age = value['Age'][0]
print('\nAppearance Age:', age)
# dec = value['Age'][0] * 10
# print('\nAppearance Age: {}-{}'.format(dec, dec + 9))








# result = {'Wrinkle Density': '0.87', 'Aging Spot': '0.5'}

# # Correlation
# data = pd.read_csv("data3.csv")
# data = data.iloc[:, 1:]
# corr = data.corr()
# print(corr['Age'].sort_values(ascending=False))


# ////////////////////////////////////////////////////////////////////////////////////////////////

# Cross Validation

# def display_scores(scores, clf):
#     print("\n" + clf.__class__.__name__ + ":")
#     print(" Scores:", scores)
#     print("Mean:", scores.mean())
#     print("Standard deviation:", scores.std())
#
# svm_clf = LinearSVC(C=1, loss="squared_hinge", max_iter=1000, random_state=42)
# svm_scores = cross_val_score(svm_clf, X_train, y_train, cv=10, scoring="accuracy")
# display_scores(svm_scores, svm_clf)

# ////////////////////////////////////////////////////////////////////////////////////////////////


# # GridSearchCV

# start_time = time.time()
# grid_algor = LinearSVC()
# param_grid = {
#     'C': list(range(1, 500, 10)),
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

# bestlinearSVC = linearSVC.best_estimator_


# ////////////////////////////////////////////////////////////////////////////////////////////////

# Counting

# k = 0
# con0 = 0
# con1 = 0
# con2 = 0
# con3 = 0
# con4 = 0
# con5 = 0
# con6 = 0
# con7 = 0
# con8 = 0
# con9 = 0
#
#
# for l in labels:
#     if l == 0:
#         con0 += 1
#     elif l == 1:
#         con1 += 1
#     elif l == 2:
#         con2 += 1
#     elif l == 3:
#         con3 += 1
#     elif l == 4:
#         con4 += 1
#     elif l == 5:
#         con5 += 1
#     elif l == 6:
#         con6 += 1
#     elif l == 7:
#         con7 += 1
#     elif l == 8:
#         con8 += 1
#     elif l == 9:
#         con9 += 1
#
# cons = [con0, con1, con2, con3, con4, con5, con6, con7, con8, con9]
#
# for l in cons:
#     if k != 100:
#         print('Count:{}-{} = {}'.format(k, k+9, l))
#     k += 10
#
# print('Whole Dataset:', count)
# print(labels)
# print(wrinkles)
#
# print('Whole Dataset:', count)

# ////////////////////////////////////////////////////////////////////////////////////////////////

# All Classes
# for row in data[1:]:
#     count += 1
#     rf = [float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6])]
#     wrinkles.append(rf)
#     labels.append(int(row[1]))
# Accuracy: 2.91%
# MSE: 480.81

# From 20 to 50
# for row in data[1:]:
#     if 50 >= int(row[1]) >= 20:
#         count += 1
#         rf = [float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6])]
#         wrinkles.append(rf)
#         labels.append(int(row[1]))
# # Accuracy: 4.20%
# # MSE: 132.43
# # count = 715 (1028)

# 3 Classes
# for row in data[1:]:
#     if int(row[1]) < 20:        # Young
#         row[1] = 0
#     elif int(row[1]) <= 50:     # Adults
#         row[1] = 1
#     elif int(row[1]) > 50:      # Old
#         row[1] = 2
#     count += 1
#     rf = [float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6])]
#     wrinkles.append(rf)
#     labels.append(int(row[1]))
# # Accuracy: 65.05%
# # MSE: 0.35

# 2 Classes
# for row in data[1:]:
#     if int(row[1]) <= 50:
#         row[1] = 0
#     elif int(row[1]) > 50:
#         row[1] = 1
#     count += 1
#     rf = [float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6])]
#     wrinkles.append(rf)
#     labels.append(int(row[1]))
# # Accuracy: 71.36 %
# # MSE: 0.29


# ////////////////////////////////////////////////////////////////////////////////////////////////

# Parameters

# grid_algor = LogisticRegression()
#
# param_grid = {
#     'C': list(range(1, 10, 1)),
#     'multi_class': ['auto'],
#     'solver': ['lbfgs'],
#     'random_state': [42]
# }
# Accuracy: 29.318
# --- 80.0690758228302 seconds ---

# grid_algor = LinearSVC()
#
# param_grid = {
#     'C': list(range(1, 100, 1)),
#     'loss': ['hinge', 'squared_hinge'],
#     'random_state': [42]

# grid_algor = RandomForestClassifier()
#
# param_grid = {
#     'n_estimators': list(range(1, 500, 10)),
#     'max_leaf_nodes': [2, 4, 6, 8, 12, 16],
#     'max_depth': [2, 4, 8],
#     'random_state': [42]
# }
# Tuned hyperparameters: {'max_depth': 8, 'max_leaf_nodes': 16, 'n_estimators': 391, 'random_state': 42}
# Accuracy: 0.31873479318734793
# --- 3494.2220554351807 seconds ---


# Tuned hyperparameters: {'n_estimators': 400}
# Accuracy: 0.2871046228710462
# --- 31.232789516448975 seconds ---





# clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, max_depth=4, n_jobs=-1, random_state=42)
# GridSearchCV
# def find_hyperparameters(X_train, y_train):
#     start_time = time.time()
#
#     grid_algor = RandomForestClassifier()
#
#     param_grid = {
#         # 'n_estimators': list(range(1, 500, 10)),
#         'n_estimators': list(range(1, 500, 1)),
#         'max_leaf_nodes': [16],
#         'max_depth': [4],
#         'random_state': [42]
#     }
#     grid_search = GridSearchCV(grid_algor, param_grid, cv=10)
#     grid_search.fit(X_train, y_train)
#
#
#     print('Tuned hyperparameters:', grid_search.best_params_)
#     print('Accuracy:', grid_search.best_score_)
#     print("--- %s seconds ---" % (time.time() - start_time))
#     # bestlinearSVC = linearSVC.best_estimator_



# # 2////////////////////////////////////////////////////////////////////////////////////////////////
#
# # Read and safe data from CSV File
# csvFile = 'data3.csv'
# data = read_csv(csvFile)
# wrinkles = []
# labels = []
# count = 0
#
# for row in data[1:]:
#     if 50 >= int(row[1]) >= 20:
#         count += 1
#         rf = [float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]),
#               float(row[7]), float(row[8]), float(row[9]), float(row[10]), float(row[11]),
#               float(row[12]), float(row[13]), float(row[14]), float(row[15]), float(row[16])]
#         wrinkles.append(rf)
#         labels.append(int(row[1]))
#
# X_train, X_test, y_train, y_test = train_test_split(wrinkles, labels, test_size=0.2, random_state=42)
#
# model2 = M
# model2.train(X_train, y_train)
# model2.predict(X_test)
# model2.accuracy(y_test)
#
# # 3////////////////////////////////////////////////////////////////////////////////////////////////
#
# # Read and safe data from CSV File
# csvFile = 'data3.csv'
# data = read_csv(csvFile)
# wrinkles = []
# labels = []
# count = 0
#
# for row in data[1:]:
#     if int(row[1]) <= 50:
#         row[1] = 0
#     elif int(row[1]) > 50:
#         row[1] = 1
#     count += 1
#     rf = [float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]),
#           float(row[7]), float(row[8]), float(row[9]), float(row[10]), float(row[11]),
#           float(row[12]), float(row[13]), float(row[14]), float(row[15]), float(row[16])]
#     wrinkles.append(rf)
#     labels.append(int(row[1]))
#
# X_train, X_test, y_train, y_test = train_test_split(wrinkles, labels, test_size=0.2, random_state=42)
#
# model3 = M
# model3.train(X_train, y_train)
# model3.predict(X_test)
# model3.accuracy(y_test)
#
# # 4////////////////////////////////////////////////////////////////////////////////////////////////
#
# # Read and safe data from CSV File
# csvFile = 'data3.csv'
# data = read_csv(csvFile)
# wrinkles = []
# labels = []
# count = 0
#
# for row in data[1:]:
#     if int(row[1]) < 20:        # Young
#         row[1] = 0
#     elif int(row[1]) <= 50:     # Adults
#         row[1] = 1
#     elif int(row[1]) > 50:      # Old
#         row[1] = 2
#     count += 1
#     rf = [float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]),
#           float(row[7]), float(row[8]), float(row[9]), float(row[10]), float(row[11]),
#           float(row[12]), float(row[13]), float(row[14]), float(row[15]), float(row[16])]
#     wrinkles.append(rf)
#     labels.append(int(row[1]))
#
# X_train, X_test, y_train, y_test = train_test_split(wrinkles, labels, test_size=0.2, random_state=42)
#
# model4 = M
# model4.train(X_train, y_train)
# model4.predict(X_test)
# model4.accuracy(y_test)

