import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import dlib
import cv2
from imutils import face_utils
import imutils
import math
import itertools

from sklearn.model_selection import train_test_split
from sklearn import decomposition
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor

class AttractivenessAnalysis():

    _LR = 0 # linear regression model
    _SVR = 1 # surpport vector regression model
    _RFR = 2 # random forest regression model
    _GPR = 3 # gaussian process regression model

    PATH_LANDMARK_MODEL = '../model/face_analysis/shape_predictor_68_face_landmarks.dat'
    PATH_FACIAL_FEATURE = '../dataset/face_analysis/features_ALL.txt'
    PATH_FACIAL_ATTRACTIVENESS_RATING = '../dataset/face_analysis/ratings.txt'

    def __init__(self, algo_index):
        '''
        To initialize a regressor for training facial attractiveness prediction model
        :param algo_index:
        '''
        if algo_index == self._LR:
            self.model = LinearRegression()
        elif algo_index == self._SVR:
            self.model = SVR()
        elif algo_index == self._RFR:
            self.model = RandomForestRegressor()
        elif algo_index == self._GPR:
            self.model = GaussianProcessRegressor()

    def _analyze_face(self, path_img):
        # detect and analyze face to extracting facial landmark, and prepare facial features
        face_detector = dlib.get_frontal_face_detector()
        lm_detector_path = self.PATH_LANDMARK_MODEL
        lm_detector = dlib.shape_predictor(lm_detector_path)
        face_img = dlib.load_rgb_image(path_img)
        orig_face_img = face_img.copy()
        face_img = imutils.resize(face_img, width=500)
        gray_face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        rects = face_detector(gray_face_img, 1)

        # detect 68 facial landmarks from the frontal face
        for (k, rect) in enumerate(rects):
            shape = lm_detector(gray_face_img, rect)
            shape = face_utils.shape_to_np(shape)

        img_landmark = shape.reshape(-1)
        img_landmark = img_landmark.reshape(1, 136)

        facial_feature = self._generate_all_features(img_landmark)
        self.facial_feature = self.pca.transform(facial_feature)

        return self.facial_feature

    def prepare_dataset(self, feature_path, rating_path):
        features = np.loadtxt(feature_path, delimiter=',')
        features = features[0:-1]
        ratings = np.loadtxt(rating_path, delimiter=',')
        features_train, features_test, ratings_train, ratings_test = train_test_split(features, ratings, test_size=0.1)
        self.pca = decomposition.PCA(n_components=20)
        self.pca.fit(features_train)
        features_train = self.pca.transform(features_train)
        features_test = self.pca.transform(features_test)

        return features_train, features_test, ratings_train, ratings_test


    def train_model(self, features_train, ratings_train):
        self.model.fit(features_train, ratings_train)

        return self.model

    def _generate_all_features(self, all_landmark_coordinates):
        a = [18, 22, 23, 27, 37, 40, 43, 46, 28, 32, 34, 36, 5, 9, 13, 49, 55, 52, 58]
        combinations = itertools.combinations(a, 4)
        i = 0
        pointIndices1 = [];
        pointIndices2 = [];
        pointIndices3 = [];
        pointIndices4 = [];

        for combination in combinations:
            pointIndices1.append(combination[0])
            pointIndices2.append(combination[1])
            pointIndices3.append(combination[2])
            pointIndices4.append(combination[3])
            i = i + 1
            pointIndices1.append(combination[0])
            pointIndices2.append(combination[2])
            pointIndices3.append(combination[1])
            pointIndices4.append(combination[3])
            i = i + 1
            pointIndices1.append(combination[0])
            pointIndices2.append(combination[3])
            pointIndices3.append(combination[1])
            pointIndices4.append(combination[2])
            i = i + 1

        return self._generate_features(pointIndices1, pointIndices2, pointIndices3, pointIndices4, all_landmark_coordinates)

    def _generate_features(self, pointIndices1, pointIndices2, pointIndices3, pointIndices4, all_landmark_coordinates):
        size = all_landmark_coordinates.shape
        self.all_features = np.zeros((size[0], len(pointIndices1)))
        for x in range(0, size[0]):
            landmark_coordinates = all_landmark_coordinates[x, :]
            #         print(len(landmarkCoordinates))
            #         print(len(allLandmarkCoordinates[x, :]))
            ratios = [];
            for i in range(0, len(pointIndices1)):
                x1 = landmark_coordinates[2 * (pointIndices1[i] - 1)]
                y1 = landmark_coordinates[2 * pointIndices1[i] - 1]
                x2 = landmark_coordinates[2 * (pointIndices2[i] - 1)]
                y2 = landmark_coordinates[2 * pointIndices2[i] - 1]

                x3 = landmark_coordinates[2 * (pointIndices3[i] - 1)]
                y3 = landmark_coordinates[2 * pointIndices3[i] - 1]
                x4 = landmark_coordinates[2 * (pointIndices4[i] - 1)]
                y4 = landmark_coordinates[2 * pointIndices4[i] - 1]

                points = [x1, y1, x2, y2, x3, y3, x4, y4]
                ratios.append(self._facial_ratio(points))
            self.all_features[x, :] = np.asarray(ratios)
        return self.all_features

    def _facial_ratio(self, points):
        x1 = points[0];
        y1 = points[1];
        x2 = points[2];
        y2 = points[3];
        x3 = points[4];
        y3 = points[5];
        x4 = points[6];
        y4 = points[7];

        dist1 = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        dist2 = math.sqrt((x3 - x4) ** 2 + (y3 - y4) ** 2)

        ratio = dist1 / dist2

        return ratio

    def predict_attractiveness(self, path_img):
        facial_features = self._analyze_face(path_img)
        prediction = self.model.predict(facial_features)
        score = round(prediction[0], 2)

        return score

# if __name__ == '__main__':
#     path_img = "../dataset/test_image/ksh.jpg"
#     attr_predictor = AttractivenessAnalysis(2)
#     feature_path = AttractivenessAnalysis.PATH_FACIAL_FEATURE
#     label_path = AttractivenessAnalysis.PATH_FACIAL_ATTRACTIVENESS_RATING
#     features_train, features_test, ratings_train, ratings_test = attr_predictor.prepare_dataset(feature_path, label_path)
#     attr_predictor.train_model(features_train, ratings_train)
#     score = attr_predictor.predict_attractiveness(path_img)
#     print("Score for S.H. Kim is: ", score)