"""
Date: 2019.07.22
Programmer: Maksym Chernozhukov
Description: Code for Aging Analytics "Prediction Age by Wrinkle Features"
"""

import cv2
import math
import numpy as np
import dlib
import pickle
import imutils
import os.path
from imutils import face_utils

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


from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier
import xgboost
import time


from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

class AgingDiagnosis:

    def __init__(self):
        self.wrinkles = []
        self.labels = []
    """
        1. Additional Functions for computing
    """

    # Sobel Filter for Detection Wrinkles
    def apply_sobel(self, img, ksize, thres=None):
        Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
        Iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
        Im = cv2.magnitude(Ix, Iy)
        if thres is not None:
            _, It = cv2.threshold(Im, thres, 1, cv2.THRESH_BINARY)
            return It
        else:
            return Im

    def wrinkle_density(self, img, threshold):
        Wa = np.sum(img >= threshold)
        Pa = img.shape[0] * img.shape[1]
        result = Wa / Pa
        return result

    def wrinkle_depth(self, img, threshold):
        Wa = img[img >= threshold]
        M = np.sum(Wa)
        result = M / (255 * len(Wa))
        return result

    def avg_skin_variance(self, img):
        M = np.sum(img)
        Pa = img.shape[0] * img.shape[1]
        result = M / (255 * Pa)
        return result

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


    """
        2. Extracting Wrinkle Features
    """

    def extract_features(self, face_image):
        predictor_path = 'shape_predictor_68_face_landmarks.dat'
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(predictor_path)
        threshold = 40

        try:
            image = cv2.imread(face_image)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 1)

            # Find 68 face landmarks
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
            wrinkled = self.apply_sobel(gray, 3)

            # Compute Wrinkle: Density, Depth, Average Skin Variance
            window = wrinkled[corner_left_eye[1]:corner_left_eye[3], corner_left_eye[0]:corner_left_eye[2]]
            left_eye_wr = self.wrinkle_density(window, threshold)
            d1 = self.wrinkle_depth(window, threshold)
            v1 = self.avg_skin_variance(window)

            window = wrinkled[corner_right_eye[1]:corner_right_eye[3], corner_right_eye[0]:corner_right_eye[2]]
            right_eye_wr = self.wrinkle_density(window, threshold)
            d2 = self.wrinkle_depth(window, threshold)
            v2 = self.avg_skin_variance(window)

            window = wrinkled[forehead[1]:forehead[3], forehead[0]:forehead[2]]
            forehead_wr = self.wrinkle_density(window, threshold)
            d3 = self.wrinkle_depth(window, threshold)
            v3 = self.avg_skin_variance(window)

            window = wrinkled[cheek_left[1]:cheek_left[3], cheek_left[0]:cheek_left[2]]
            cheek_left_wr = self.wrinkle_density(window, threshold)
            d4 = self.wrinkle_depth(window, threshold)
            v4 = self.avg_skin_variance(window)

            window = wrinkled[cheek_right[1]:cheek_right[3], cheek_right[0]:cheek_right[2]]
            cheek_right_wr = self.wrinkle_density(window, threshold)
            d5 = self.wrinkle_depth(window, threshold)
            v5 = self.avg_skin_variance(window)

            # Skip NaN
            if math.isnan(left_eye_wr) or math.isnan(right_eye_wr) or math.isnan(forehead_wr) or \
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

            # Wrinkle Features
            result = [[left_eye_wr, right_eye_wr, forehead_wr, cheek_left_wr, cheek_right_wr,
                       d1, d2, d3, d4, d5,
                       v1, v2, v3, v4, v5]]

            return result
        except:
            # In some images can not detect face landmarks or compute features
            print('Try to use another picture')

    """
        3. Main Method
    """

    def make_preprocessing(self, csvFile):
        data = self.read_csv(csvFile)
        # Rescaling Size and Count of Aging Spots

        for row in data[1:]:
            rf = [float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]),
                  float(row[7]), float(row[8]), float(row[9]), float(row[10]), float(row[11]),
                  float(row[12]), float(row[13]), float(row[14]), float(row[15]), float(row[16])]

            self.wrinkles.append(rf)
            self.labels.append(int(row[1]))

    def seperate_dataset(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.wrinkles, self.labels,
                                                                                test_size=0.2, random_state=42)

        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_model(self):
        self.clf = LinearSVC(C=31, loss="squared_hinge", max_iter=1000, random_state=42)
        self.clf.fit(self.X_train, self.y_train)

        return self.clf

    def estimate_model(self):
        y_pred = self.clf.predict(self.X_test)
        accuracy = (metrics.accuracy_score(self.y_test, y_pred)) * 100.0
        print('Accuracy: {:.2f}%'.format(accuracy))

    def analyse_factors(self, face):
        features = self.extract_features(face)
        age_pred = self.trained_model.predict(features)

        # Age: 1, Features: 15    (Total 16)
        result = {'Age': age_pred[0],
                  "Density_Corner_Left_Eye": features[0][0],
                  "Density_Corner_Right_Eye": features[0][1],
                  "Density_Forehead": features[0][2],
                  "Density_Cheek_Left": features[0][3],
                  "Density_Cheek_Right": features[0][4],

                  "Depth_Corner_Left_Eye": features[0][5],
                  "Depth_Corner_Right_Eye": features[0][6],
                  "Depth_Forehead": features[0][7],
                  "Depth_Cheek_Left": features[0][8],
                  "Depth_Cheek_Right": features[0][9],

                  "Variance_Corner_Left_Eye": features[0][10],
                  "Variance_Corner_Right_Eye": features[0][11],
                  "Variance_Forehead": features[0][12],
                  "Variance_Cheek_Left": features[0][13],
                  "Variance_Cheek_Right": features[0][14]}

        return result


if __name__ == '__main__':
    csvFile = './data/data7.csv'
    face_path = '../datasets/experement/max1.jpg'
    model_path = './trained_models/trained_model8.pkl'

    aging_diag = AgingDiagnosis()
    # Make trained model, estimate and save the model in a file
    aging_diag.make_preprocessing(csvFile)
    X_train, X_test, y_train, y_test = aging_diag.seperate_dataset()

    # aging_diag.train_model()
    # aging_diag.estimate_model()
    # aging_diag.save_model(model_path)


    # Run Analysing

    # aging_diag.read_model(model_path)                            # Activate the trained model
    # result = aging_diag.analyse_factors(face_path)               # Analyse factors

    # Show Results
    # age = result['Age']

    def display_scores(scores, clf):
        print("\n" + clf.__class__.__name__ + ":")
        print(" Scores:", scores)
        print("Mean:", scores.mean())
        print("Standard deviation:", scores.std())

    log_clf = LogisticRegression(C=1, multi_class="auto", solver="lbfgs", max_iter=1000, random_state=42)
    svm_clf = LinearSVC(C=31, loss="squared_hinge", max_iter=1000, random_state=42)
    rnd_clf = RandomForestClassifier(n_estimators=236, max_leaf_nodes=16,
                                     max_depth=4, n_jobs=-1, random_state=42)
    knn_clf = KNeighborsClassifier(n_neighbors=8, metric='euclidean')

    # voting_clf = VotingClassifier(
    #     estimators=[('lr', log_clf), ('svm', svm_clf), ('rf', rnd_clf), ('knn', knn_clf)],
    #     voting='hard')
    # voting_clf.fit(X_train, y_train)
    #
    # voting_clf_soft = VotingClassifier(
    #     estimators=[('lr', log_clf), ('svm', svm_clf), ('rf', rnd_clf), ('knn', knn_clf)],
    #     voting='soft')
    # voting_clf_soft.fit(X_train, y_train)
    #
    # ada_clf = AdaBoostClassifier(
    #     DecisionTreeClassifier(max_depth=1), n_estimators=200,
    #     algorithm="SAMME", learning_rate=0.5)
    # ada_clf.fit(X_train, y_train)
    #
    # xgb_reg = xgboost.XGBRegressor()
    # xgb_reg.fit(X_train, y_train)

    # for clf in (log_clf, svm_clf, rnd_clf, knn_clf, voting_clf, voting_clf_soft, ada_clf, xgb_reg):
    for clf in (log_clf, svm_clf):
        svm_scores = cross_val_score(clf, X_train, y_train, cv=10, scoring="accuracy")
        display_scores(svm_scores, clf)

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(clf.__class__.__name__, accuracy_score(y_test, y_pred))


# LogisticRegression:
#  Scores: [0.02471679 0.02919708 0.02523659 0.02653928 0.03783784 0.02508179
#  0.03516484 0.02330744 0.04026846 0.03837472]
# Mean: 0.030572481013531227
# Standard deviation: 0.00626585870238631
# LogisticRegression: 0.037149028077753776
# LogisticRegression 0.035853131749460046

# LinearSVC:
# Scores: [0.03604531 0.03858186 0.03364879 0.03503185 0.03567568 0.02508179
#              0.03186813 0.02108768 0.03355705 0.02934537]
# Mean: 0.031992350386950856
# Standard deviation: 0.0051140513639105906
# LinearSVC 0.031965442764578834


# RandomForestClassifier:
# Scores: [0.03810505 0.03441084 0.04206099 0.04883227 0.03243243 0.03816794
#                   0.03846154 0.0399556  0.03579418 0.03047404]
# Mean: 0.037869488995521
# Standard deviation: 0.004930736149553334
# RandomForestClassifier 0.031965442764578834

    # F1 = result['Density_Corner_Left_Eye']
    # F2 = result['Density_Corner_Right_Eye']
    # F3 = result['Density_Forehead']
    # F4 = result['Density_Cheek_Left']
    # F5 = result['Density_Cheek_Right']
    #
    # F6 = result['Depth_Corner_Left_Eye']
    # F7 = result['Depth_Corner_Right_Eye']
    # F8 = result['Depth_Forehead']
    # F9 = result['Depth_Cheek_Left']
    # F10 = result['Depth_Cheek_Right']
    #
    # F11 = result['Variance_Corner_Left_Eye']
    # F12 = result['Variance_Corner_Right_Eye']
    # F13 = result['Variance_Forehead']
    # F14 = result['Variance_Cheek_Left']
    # F15 = result['Variance_Cheek_Right']
    #
    # print('\nDensity F1:', F1)
    # print('Density F2:', F2)
    # print('Density F3:', F3)
    # print('Density F4:', F4)
    # print('Density F5:', F5)
    # print('\nDepth F6:', F6)
    # print('Depth F7:', F7)
    # print('Depth F8:', F8)
    # print('Depth F9:', F9)
    # print('Depth F10:', F10)
    # print('\nVariance F11:', F11)
    # print('Variance F12:', F12)
    # print('Variance F13:', F13)
    # print('Variance F14:', F14)
    # print('Variance F15:', F15)

    # # Show Image
    # img = cv2.imread(face_path)
    # img = imutils.resize(img, width=500)
    # cv2.imshow("Face Image", img)
    # cv2.waitKey(0)

