"""
Date: 2019.07.24
Programmer: Maksym Chernozhukov
Description: Code for Aging Analytics "Prediction Age by Aging Spot Features"
"""

import cv2
import math
import numpy as np
import pandas as pd
import dlib
import pickle
import imutils
import os.path
import time
from threading import Thread
from imutils import face_utils
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler


class AgingDiagnosis:

    """
        1. Additional Functions
    """

    # Sobel Filter for Detection Spots
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
        cd = 100.0       # Max count of spots
        try:
            for s in spots[0][0:5]:
                sum += s
                count += 1
            density = float(sum) / float(count)
        except:
            density = 0.0

        for s in spots[0][5:10]:
            if s > spot_size:
                spot_size = s

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

    def open_model(self, pickle_path):
        file_r = open(pickle_path, 'rb')
        self.trained_model = pickle.load(file_r)
        return self.trained_model

    """
        2. Extracting Wrinkle Features
    """

    def extract_features(self, image):
        predictor_path = './models/shape_predictor_68_face_landmarks.dat'
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(predictor_path)
        threshold = 40
        cd = 100.0      # For 'Spots Count' to make value [0:1]
        Sn_size = []

        try:
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

            # Compute Spots
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
            return None

    """
        3. Main Methods
    """

    def analyse_factors(self, face):
        def th(face):
            try:
                features = self.extract_features(face)
                if features != None:
                    y_pred = self.trained_model.predict(features)
                    sp_values = self.calculate_spot_value(features)

                    # Age: 1, Features: 15, Summarize Features: 1 ([Mean Density, Max Spot Size, Total Count Spots]). (Total 17)
                    self.result = {'Age': y_pred[0],
                              "Spot_Density_Corner_Left_Eye": features[0][0],
                              "Spot_Density_Corner_Right_Eye": features[0][1],
                              "Spot_Density_Forehead": features[0][2],
                              "Spot_Density_Cheek_Left": features[0][3],
                              "Spot_Density_Cheek_Right": features[0][4],

                              "Max_Spot_Size_Corner_Left_Eye": features[0][5],
                              "Max_Spot_Size_Corner_Right_Eye": features[0][6],
                              "Max_Spot_Size_Forehead": features[0][7],
                              "Max_Spot_Size_Cheek_Left": features[0][8],
                              "Max_Spot_Size_Cheek_Right": features[0][9],

                              "Count_of_Spots_Corner_Left_Eye": int(features[0][10]*100),  # convert '0,03' to '3' spots
                              "Count_of_Spots_Corner_Right_Eye": int(features[0][11]*100),
                              "Count_of_Spots_Forehead": int(features[0][12]*100),
                              "Count_of_Spots_Cheek_Left": int(features[0][13]*100),
                              "Count_of_Spots_Cheek_Right": int(features[0][14]*100),
                              'Aging_Spot_Features_Total': sp_values}
                else:
                    self.result = None
            except:
                self.result = None
        t = Thread(target=th, args=(face,))
        t.start()

        def get_result(self):
            return self.result


if __name__ == '__main__':

    # Select Paths
    face_path = '../datasets/experement/spots.jpg'
    model_path = './trained_models/aging_spot_model.pkl'
    image = cv2.imread(face_path)

    aging_diag = AgingDiagnosis()
    aging_diag.open_model(model_path)                       # Activate trained model
    result = aging_diag.analyse_factors(image)          # Analyse factors

    if result == None:
        print('There is ERROR')
    else:
        # Show Results
        age = result['Age']
        spots_values = result['Aging_Spot_Features_Total']

        F1 = result['Spot_Density_Corner_Left_Eye']
        F2 = result['Spot_Density_Corner_Right_Eye']
        F3 = result['Spot_Density_Forehead']
        F4 = result['Spot_Density_Cheek_Left']
        F5 = result['Spot_Density_Cheek_Right']

        F6 = result['Max_Spot_Size_Corner_Left_Eye']
        F7 = result['Max_Spot_Size_Corner_Right_Eye']
        F8 = result['Max_Spot_Size_Forehead']
        F9 = result['Max_Spot_Size_Cheek_Left']
        F10 = result['Max_Spot_Size_Cheek_Right']

        F11 = result['Count_of_Spots_Corner_Left_Eye']
        F12 = result['Count_of_Spots_Corner_Right_Eye']
        F13 = result['Count_of_Spots_Forehead']
        F14 = result['Count_of_Spots_Cheek_Left']
        F15 = result['Count_of_Spots_Cheek_Right']

        print('\nAppearance Age:', age)
        print('\nSpots Density: {:.6f}%'.format(spots_values[0]))
        print('Max Spots Size: {:.6f}'.format(spots_values[1]))
        print('Count of Spots: {}'.format(spots_values[2]))

        print('\nDensity F1:', F1)
        print('Density F2:', F2)
        print('Density F3:', F3)
        print('Density F4:', F4)
        print('Density F5:', F5)
        print('\nSize F6:', F6)
        print('Size F7:', F7)
        print('Size F8:', F8)
        print('Size F9:', F9)
        print('Size F10:', F10)
        print('\nCount F11:', F11)
        print('Count F12:', F12)
        print('Count F13:', F13)
        print('Count F14:', F14)
        print('Count F15:', F15)

        # Show Image
        img = cv2.imread(face_path)
        img = imutils.resize(img, width=500)
        cv2.imshow("Face Image", img)
        cv2.waitKey(0)
