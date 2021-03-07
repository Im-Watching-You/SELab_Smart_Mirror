"""
Date: 2019.07.24
Programmer: Maksym Chernozhukov
Description: Code for Aging Analytics "Prediction Age by Wrinkle and Aging Spot Features"
"""

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

class AgingDiagnosis:
    def __init__(self):
        self.wrinkles = []
        self.labels = []
    """
        1. Additional Functions
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

    def calculate_spot_value(self, spots):
        sum = 0
        count = 0
        spot_size = 0
        count_spots = 0
        cd = 100.0       # max count of spots in dataset

        # for s in spots[0][0:2]:
        for s in spots[0][0:5]:
            sum += s
            count += 1
        density = float(sum) / float(count)

        # for s in spots[0][2:4]:
        for s in spots[0][5:10]:
            if s > spot_size:
                spot_size = s

        # for s in spots[0][4:7]:
        for s in spots[0][10:15]:
            count_spots += s

        result = [density, spot_size, int(count_spots*cd)]
        return result

    def detect_blobs(self, img):
        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()

        # Filter by Area.
        params.filterByArea = True
        params.minArea = 30
        params.maxArea = 5000

        # Filter by Color
        params.filterByColor = True
        params.blobColor = 0

        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.5

        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.7

        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.2

        # Create a detector with the parameters
        ver = (cv2.__version__).split('.')
        if int(ver[0]) < 3:
            detector = cv2.SimpleBlobDetector(params)
        else:
            detector = cv2.SimpleBlobDetector_create(params)

        # Detect blobs.
        keypoints = detector.detect(img)
        return keypoints

    def extract_spots(self, window, threshold):
        Sa = 0
        spots_size = 0
        coordinates = []

        # Image Preprocecing
        img = (window).astype('uint8')
        img[img <= threshold] = 255
        img[img != 255] = 0

        # Find Coordinates of Spots
        keypoints = self.detect_blobs(img)
        for k in keypoints:
            coordinates.append(
                [int(k.pt[0] - k.size * 0.5), int(k.pt[1] - k.size * 0.5), int(k.size) + 1, int(k.size) + 1])
            if int(k.size) > spots_size:
                spots_size = int(k.size) + 1

        # Coordinates = [[x, y, w, h], ...]
        for cour in coordinates:
            spot = img[cour[1]: cour[1] + cour[3], cour[0]: cour[0] + cour[2]]
            s = np.sum(spot == 0)
            Sa += s

        Pa = img.shape[0] * img.shape[1]
        spots_count = len(coordinates)
        density = float(Sa) / float(Pa)

        result = [density, spots_size, spots_count]
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

    def open_model(self, pickle_path):
        file_r = open(pickle_path, 'rb')
        self.trained_model = pickle.load(file_r)
        return self.trained_model

    """
        2. Extracting Wrinkle Features
    """

    def extract_features(self, face):
        predictor_path = 'shape_predictor_68_face_landmarks.dat'
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(predictor_path)
        threshold = 40
        cd = 100.0

        try:
            image = cv2.imread(face)
            # image = imutils.resize(image, width=500)
            width = image.shape[1]
            height = image.shape[0]
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
            wrinkled = self.apply_sobel(gray, 3)

            # Compute Aging Spots
            window = wrinkled[corner_left_eye[1]:corner_left_eye[3], corner_left_eye[0]:corner_left_eye[2]]
            s1 = self.extract_spots(window, threshold)

            window = wrinkled[corner_right_eye[1]:corner_right_eye[3], corner_right_eye[0]:corner_right_eye[2]]
            s2 = self.extract_spots(window, threshold)

            window = wrinkled[forehead[1]:forehead[3], forehead[0]:forehead[2]]
            s3 = self.extract_spots(window, threshold)

            window = wrinkled[cheek_left[1]:cheek_left[3], cheek_left[0]:cheek_left[2]]
            s4 = self.extract_spots(window, threshold)

            window = wrinkled[cheek_right[1]:cheek_right[3], cheek_right[0]:cheek_right[2]]
            s5 = self.extract_spots(window, threshold)

            # Convert Max Spot Size (px) to Rate
            S_size = [s1[1], s2[1], s3[1], s4[1], s5[1]]
            Sn_size = []
            for s in S_size:
                if s != 0:
                    rate = (float(s) * float(s)) / (float(width) * float(height))
                    Sn_size.append(rate)
                else:
                    Sn_size.append(0.0)

            result = [[s1[0], s2[0], s3[0], s4[0], s5[0],
                       Sn_size[0], Sn_size[1], Sn_size[2], Sn_size[3], Sn_size[4],
                       float(s1[2])/cd, float(s2[2])/cd, float(s3[2])/cd, float(s4[2])/cd, float(s5[2])/cd]]

            return result
        except:
            print('Try to use another picture')

    """
        3. Main Methods
    """

    def make_preprocessing(self, csvFile):
        data = self.read_csv(csvFile)
        # Rescaling Size and Count of Aging Spots
        cd = 100.0

        for row in data[1:]:
            rf = [float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]),
                  float(row[7]), float(row[8]), float(row[9]), float(row[10]), float(row[11]),
                  float(row[12]) / cd, float(row[13]) / cd, float(row[14]) / cd, float(row[15]) / cd,
                  float(row[16]) / cd]

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

    def analyse_factors(self, face):
        f_spots = self.extract_features(face)
        y_pred = self.trained_model.predict(f_spots)
        sp_value = self.calculate_spot_value(f_spots)

        result = {'Age': y_pred[0], 'Aging Spot Features': sp_value}
        return result


if __name__ == '__main__':
    csvFile = './data/data6_spots.csv'
    face_path = '../datasets/experement/max1.jpg'
    model_path = './trained_models/trained_model6.pkl'

    aging_diag = AgingDiagnosis()

    # Make trained model, estimate and save the model in a file
    aging_diag.make_preprocessing(csvFile)
    # aging_diag.seperate_dataset()
    #
    # aging_diag.train_model()
    # aging_diag.estimate_model()
    # aging_diag.save_model(model_path)
    #
    # Run Analysing
    aging_diag.open_model(model_path)                       # Activate trained model
    result = aging_diag.analyse_factors(face_path)          # Analyse factors

    # Show Results
    age = result['Age']
    spots_value = result['Aging Spot Features']

    print('\nAppearance Age:', age)
    print('\nSpots Density: {:.4f}%'.format(spots_value[0]))
    print('Max Spots Size: {:.4f}'.format(spots_value[1]))
    print('Count of Spots: {}'.format(spots_value[2]))

# Show Image
# img = cv2.imread(face_path)
# img = imutils.resize(img, width=500)
# cv2.imshow("Face Image", img)
# cv2.waitKey(0)

# //////////////////////////////////////////////////////////
# Accuracy: 3.02% Default

# Notes

# boy1 = 10
# girl1 = 15
# girl2 = 15
# girl3 = 39
# man1 = 20
# man2 = 20
# max1 = 26
# max2 = 26
# max3 = 26
# max4 = 26
# mk1 = 27
# mk2 = 27
# mk3 = 27
# kr = 25
# old1 = 54
# old2 = 66
# old3 = 80
# old4 = 87
# old5 = 93
# super_old1 = 112
# super_old2 = 104

# # Correlation
# data = pd.read_csv("./data/data3.csv")
# data = data.iloc[:, 1:]
# corr = data.corr()
# print(corr['Age'].sort_values(ascending=False))


# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))


# Result for Case 5
# dec = result['Age'][0] * 10
# print('\nAppearance Age: {}-{}'.format(dec, dec + 9))


# ////////////////////////////////////////////////////////////////////////////////////////////

# # Small Dataset (Only with Spots) With Correlated Features
# result = [[s2[0], s3[0],                                                # D: CRY, F
#            Sn_size[1], Sn_size[4],                                      # M: CRY, CR
#            float(s2[2]) / cd, float(s3[2]) / cd, float(s4[2]) / cd]]    # C: CRY, F, CL

# # Huge Dataset With Correlated Features
# result = [[s3[0], s4[0], s5[0],                                 # D: F, CL, CR
#            Sn_size[0], Sn_size[3], Sn_size[4],                  # M: CLY, CL, CR
#            float(s3[2])/cd, float(s4[2])/cd, float(s5[2])/cd]]  # C: F, CL, CR


# # Small Dataset (Only with Spots) With Correlated Features
# rf = [float(row[3]), float(row[4]),                                   # D: CRY, F
#       float(row[8]), float(row[11]),                                  # M: CRY, CR
#       float(row[13]) / cd, float(row[14]) / cd, float(row[15]) / cd]  # C: CRY, F, CL

# # Huge Dataset With Correlated Features
# rf = [float(row[4]), float(row[5]), float(row[6]),                      # D: F, CL, CR
#       float(row[7]), float(row[10]), float(row[11]),                    # M: CLY, CL, CR
#       float(row[14]) / cd, float(row[15]) / cd, float(row[16]) / cd]    # D: F, CL, CR

