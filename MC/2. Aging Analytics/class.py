"""
Date: 2019.07.29
Programmer: Maksym Chernozhukov
Description: Code for Aging Analytics "Prediction Age by Wrinkle, Aging Spot, Distance Features"
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

class AgingDiagnosis:
    def __init__(self):
        self.wrinkles = []
        self.labels = []

        predictor_path68 = 'shape_predictor_68_face_landmarks.dat'
        self.predictor68 = dlib.shape_predictor(predictor_path68)
        predictor_path81 = "shape_predictor_81_face_landmarks.dat"
        self.predictor81 = dlib.shape_predictor(predictor_path81)
        self.model_path = './trained_models/trained_model7.pkl'
        self.detector = dlib.get_frontal_face_detector()
        self.age_predicter = joblib.load(self.model_path)

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

    # Mean Value of Wrinkles
    def calculate_wrinkle_value(self, wrinkle):
        sum = 0
        count = 0
        for w in wrinkle[0]:
            sum += w
            count += 1
        result = float(sum) / float(count)
        return result

    # Calculate Mean Value of Aging Spots; Find Max Size Aging Spot; Total Count of Spots
    def calculate_spot_value(self, spots):
        sum = 0
        count = 0
        spot_size = 0
        count_spots = 0
        cd = 100.0  # max count of spots in dataset

        for s in spots[0][0:5]:
            sum += s
            count += 1
        density = float(sum) / float(count)

        for s in spots[0][5:10]:
            if s > spot_size:
                spot_size = s

        for s in spots[0][10:15]:
            count_spots += s

        result = [density, spot_size, int(count_spots * cd)]
        return result

    def compute_distance(self, p1, p2):
        distance = math.sqrt(((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2))
        return distance

    def compute_middle(self, p1, p2):
        p3 = ((p2[0] + p1[0]) / 2, (p2[1] + p1[1]) / 2)
        return p3

    def compute_area(self, w, h, type):
        result = 0
        if type=="Square":
            result= w*h
        elif type == "Triangle":
            result = w*h/2
        return result

    def extract_face_landmark(self, face_path):
        img = cv2.imread(face_path)
        d = self.detector(img, 0)
        shape = self.predictor81(img, d[0])  # find facial landmarks
        return shape

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

    """
        2. Extracting Wrinkle, Aging Spots, ... Features
    """

    def compute_factors(self, shape, w, h):
        result = {"F1": round(self.compute_distance(shape[45], shape[36]) / w, 2),
                  "F2": round(self.compute_distance(shape[42], shape[39]) / w, 2),
                  "F3": round(self.compute_distance(self.compute_middle(shape[46], shape[43]),
                                                    self.compute_middle(shape[40], shape[37])) / w, 2),
                  "F4": round(self.compute_distance(shape[44], shape[43]) / w, 2),
                  "F5": round(self.compute_distance(shape[35], shape[31]) / w, 2),
                  "F6": round(self.compute_distance(shape[54], shape[48]) / w, 2),
                  "F7": round(self.compute_distance(shape[27], shape[33]) / h, 2),
                  "F8": round(self.compute_distance(shape[27], shape[8]) / h, 2),
                  "F9": round(self.compute_distance(shape[33], shape[8]) / h, 2),
                  "F10": round(self.compute_distance(shape[33], shape[27]) / h, 2),
                  'F11': round(self.compute_distance(shape[33], shape[51]) / h, 2),
                  'F12': round(self.compute_area(self.compute_distance(shape[27], shape[33]),
                                                 self.compute_distance(shape[31], shape[35]), "Triangle")
                               / self.compute_area(w, h, "Square"), 2)}

        return result

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
                spots_size = int(k.size) + 1  # Calculate Size of Spots

        # Coordinates = [[x, y, w, h], ...]
        for cour in coordinates:
            spot = img[cour[1]: cour[1] + cour[3], cour[0]: cour[0] + cour[2]]
            s = np.sum(spot == 0)
            Sa += s

        # Calculate Density and Counts of Spots
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

    def extract_features(self, face):
        threshold = 40
        cd = 100.0

        try:
            image = cv2.imread(face)
            width = image.shape[1]
            height = image.shape[0]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rects = self.detector(gray, 1)

            shape = self.predictor68(gray, rects[0])
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

            # Compute Wrinkles
            window = wrinkled[corner_left_eye[1]:corner_left_eye[3], corner_left_eye[0]:corner_left_eye[2]]
            left_eye_wr = self.wrinkle_density(window, threshold)
            d1 = self.wrinkle_depth(window, threshold)
            v1 = self.avg_skin_variance(window)
            s1 = self.extract_spots(window, threshold)

            window = wrinkled[corner_right_eye[1]:corner_right_eye[3], corner_right_eye[0]:corner_right_eye[2]]
            right_eye_wr = self.wrinkle_density(window, threshold)
            d2 = self.wrinkle_depth(window, threshold)
            v2 = self.avg_skin_variance(window)
            s2 = self.extract_spots(window, threshold)

            window = wrinkled[forehead[1]:forehead[3], forehead[0]:forehead[2]]
            forehead_wr = self.wrinkle_density(window, threshold)
            d3 = self.wrinkle_depth(window, threshold)
            v3 = self.avg_skin_variance(window)
            s3 = self.extract_spots(window, threshold)

            window = wrinkled[cheek_left[1]:cheek_left[3], cheek_left[0]:cheek_left[2]]
            cheek_left_wr = self.wrinkle_density(window, threshold)
            d4 = self.wrinkle_depth(window, threshold)
            v4 = self.avg_skin_variance(window)
            s4 = self.extract_spots(window, threshold)

            window = wrinkled[cheek_right[1]:cheek_right[3], cheek_right[0]:cheek_right[2]]
            cheek_right_wr = self.wrinkle_density(window, threshold)
            d5 = self.wrinkle_depth(window, threshold)
            v5 = self.avg_skin_variance(window)
            s5 = self.extract_spots(window, threshold)

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

            # Convert Max Spot Size (px) to Rate
            S_size = [s1[1], s2[1], s3[1], s4[1], s5[1]]
            Sn_size = []
            for s in S_size:
                if s != 0:
                    rate = (float(s) * float(s)) / (float(width) * float(height))
                    Sn_size.append(rate)
                else:
                    Sn_size.append(0.0)

            # Wrinkle Features
            result_wr = [[left_eye_wr, right_eye_wr, forehead_wr, cheek_left_wr, cheek_right_wr,
                          d1, d2, d3, d4, d5,
                          v1, v2, v3, v4, v5]]

            # Aging Spot Features
            result_sp = [[s1[0], s2[0], s3[0], s4[0], s5[0],
                          Sn_size[0], Sn_size[1], Sn_size[2], Sn_size[3], Sn_size[4],
                          float(s1[2]) / cd, float(s2[2]) / cd, float(s3[2]) / cd, float(s4[2]) / cd,
                          float(s5[2]) / cd]]

            # Wrinkle and Aging Spot Features
            all = [[left_eye_wr, right_eye_wr, forehead_wr, cheek_left_wr, cheek_right_wr,
                    d1, d2, d3, d4, d5,
                    v1, v2, v3, v4, v5,
                    s1[0], s2[0], s3[0], s4[0], s5[0],
                    Sn_size[0], Sn_size[1], Sn_size[2], Sn_size[3], Sn_size[4],
                    float(s1[2]) / cd, float(s2[2]) / cd, float(s3[2]) / cd, float(s4[2]) / cd, float(s5[2]) / cd]]

            return all, result_wr, result_sp
        except:
            print('Try to use another picture')

    """
            3. Main Method
    """
    def compute_age_factors(self, face_path):
        shape = self.extract_face_landmark(face_path)
        mtx_landmarks = np.matrix([[p.x, p.y] for p in shape.parts()])
        landmarks = np.squeeze(np.asarray(mtx_landmarks))
        width = self.compute_distance(landmarks[15], landmarks[1])
        height = self.compute_distance(landmarks[8], (landmarks[1][0], min(landmarks[72][1], landmarks[69][1])))
        r = self.compute_factors(landmarks, width, height)
        features = np.reshape(list(r.values()), (1, -1))

        features_list = []
        for f in features[0]:
            features_list.append(f)

        return features_list

    def analyse_factors(self, face_path):
        features_mh = self.compute_age_factors(face_path)
        features_mc, f_wrinkles, f_spots = self.extract_features(face_path)
        print(features_mc)
        print(type(features_mc))
        features = [features_mc[0] + features_mh]
        print(features)
        age_pred = self.age_predicter.predict(features)
        wr_value = self.calculate_wrinkle_value(f_wrinkles)
        sp_value = self.calculate_spot_value(f_spots)

        result = {'Age': age_pred[0],
                  "Density_Corner_Left_Eye": f_wrinkles[0][0],
                  "Density_Corner_Right_Eye": f_wrinkles[0][1],
                  "Density_Forehead": f_wrinkles[0][2],
                  "Density_Cheek_Left": f_wrinkles[0][3],
                  "Density_Cheek_Right": f_wrinkles[0][4],

                  "Depth_Corner_Left_Eye": f_wrinkles[0][5],
                  "Depth_Corner_Right_Eye": f_wrinkles[0][6],
                  "Depth_Forehead": f_wrinkles[0][7],
                  "Depth_Cheek_Left": f_wrinkles[0][8],
                  "Depth_Cheek_Right": f_wrinkles[0][9],

                  "Variance_Corner_Left_Eye": f_wrinkles[0][10],
                  "Variance_Corner_Right_Eye": f_wrinkles[0][11],
                  "Variance_Forehead": f_wrinkles[0][12],
                  "Variance_Cheek_Left": f_wrinkles[0][13],
                  "Variance_Cheek_Right": f_wrinkles[0][14],

                  "Spot_Density_Corner_Left_Eye": f_spots[0][0],
                  "Spot_Density_Corner_Right_Eye": f_spots[0][1],
                  "Spot_Density_Forehead": f_spots[0][2],
                  "Spot_Density_Cheek_Left": f_spots[0][3],
                  "Spot_Density_Cheek_Right": f_spots[0][4],

                  "Max_Spot_Size_Corner_Left_Eye": f_spots[0][5],
                  "Max_Spot_Size_Corner_Right_Eye": f_spots[0][6],
                  "Max_Spot_Size_Forehead": f_spots[0][7],
                  "Max_Spot_Size_Cheek_Left": f_spots[0][8],
                  "Max_Spot_Size_Cheek_Right": f_spots[0][9],

                  "Count_of_Spots_Corner_Left_Eye": int(f_spots[0][10] * 100),  # convert '0,03' to '3' spots
                  "Count_of_Spots_Corner_Right_Eye": int(f_spots[0][11] * 100),
                  "Count_of_Spots_Forehead": int(f_spots[0][12] * 100),
                  "Count_of_Spots_Cheek_Left": int(f_spots[0][13] * 100),
                  "Count_of_Spots_Cheek_Right": int(f_spots[0][14] * 100),

                  'Wrinkle_Features': wr_value,
                  'Aging_Spot_Features': sp_value}

        return result

if __name__ == '__main__':
    face_path = '../datasets/experement/old3.jpg'
    # face_path = '../datasets/wrinkles3/10023_1947-8-22_1976.jpg'


    aging_diag = AgingDiagnosis()
    result = aging_diag.analyse_factors(face_path)

    # Show Results
    age = result['Age']
    wrinkle_value = result['Wrinkle_Features']
    spots_value = result['Aging_Spot_Features']

    print('\nAppearance Age:', age)
    print('Wrinkles: {:.2f}%'.format(wrinkle_value * 100.0))
    print('\nSpots Density: {:.6f}%'.format(spots_value[0]))
    print('Max Spots Size: {}'.format(spots_value[1]))
    print('Count of Spots: {}'.format(spots_value[2]))

    # # # Correlation
    # data = pd.read_csv("./data/data7.csv")
    # data = data.iloc[:, 1:]
    # corr = data.corr()
    # print(corr['Age'].sort_values(ascending=False))
    #
    # # Show Image
    # img = cv2.imread(face_path)
    # img = imutils.resize(img, width=500)
    # cv2.imshow("Face Image", img)
    # cv2.waitKey(0)

