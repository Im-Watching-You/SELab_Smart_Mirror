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
import time
from threading import Thread
from imutils import face_utils


class AgingDiagnosis:

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

    def set_model(self, trained_model):
        self.trained_model = trained_model

    def wrinkle_density(self, img, threshold):
        try:
            Wa = np.sum(img >= threshold)
            Pa = img.shape[0] * img.shape[1]
            result = Wa / Pa
            return result
        except:
            print('Wrinkle Density: None')
            return None

    def wrinkle_depth(self, img, threshold):
        try:
            Wa = img[img >= threshold]
            M = np.sum(Wa)
            result = M / (255 * len(Wa))
            return result
        except:
            print('Wrinkle Depth: None')
            return None

    def avg_skin_variance(self, img):
        try:
            M = np.sum(img)
            Pa = img.shape[0] * img.shape[1]
            result = M / (255 * Pa)
            return result
        except:
            print('Wrinkle Variance: None')
            return None

    def open_model(self, pickle_path):
        file_r = open(pickle_path, 'rb')
        self.trained_model = pickle.load(file_r)
        return self.trained_model

    def calculate_wrinkle_value(self, wrinkles):
        try:
            sum = 0
            count = 0
            print(wrinkles)
            for f in wrinkles[0]:
                sum += float(f)
                count += 1
            result = float(sum) / float(count)
            return round(result,2)
        except:
            return 0.0
    """
        2. Extracting Wrinkle Features
    """

    def extract_features(self, face_image):
        predictor_path = './models/shape_predictor_68_face_landmarks.dat'
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(predictor_path)
        threshold = 40

        try:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
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

            # Corner Left Eye, Corner Right Eye, Forehead Values > 0.8 generally Error (hair, sunglasses, and etc.)
            if left_eye_wr > 0.8:
                left_eye_wr = (cheek_left_wr + cheek_right_wr) * 0.5
            if right_eye_wr > 0.8:
                right_eye_wr = (cheek_left_wr + cheek_right_wr) * 0.5
            if forehead_wr > 0.8:
                forehead_wr = (cheek_left_wr + cheek_right_wr) * 0.5

            if d1 != None and d2 != None and d3 != None and d4 != None and d5 != None:
                result = {"Density_Corner_Left_Eye": left_eye_wr,
                          "Density_Corner_Right_Eye": right_eye_wr,
                          "Density_Forehead": forehead_wr,
                          "Density_Cheek_Left": cheek_left_wr,
                          "Density_Cheek_Right": cheek_right_wr,
                          "Depth_Corner_Left_Eye": d1,
                          "Depth_Corner_Right_Eye": d2,
                          "Depth_Forehead": d3,
                          "Depth_Cheek_Left": d4,
                          "Depth_Cheek_Right": d5,
                          "Variance_Corner_Left_Eye": v1,
                          "Variance_Corner_Right_Eye": v2,
                          "Variance_Forehead": v3,
                          "Variance_Cheek_Left": v4,
                          "Variance_Cheek_Right": v5}
            else:
                result = None

            return result
        except:
            # In some images can not detect face landmarks or compute features
            print('Try to use another picture')
            return None

    """
        3. Main Method
    """

    def analyse_factors(self, face):
        def th(face):
            features = self.extract_features(face)
            if features != None:
                f = [list(features.values())]
                y_pred = self.trained_model.predict(f)
                value = self.calculate_wrinkle_value(f)

                self.result = {'Age': y_pred[0], 'Wrinkle Features': value, "Whole Features":features}
            else:
                self.result = None
        t = Thread(target=th, args=(face,))
        t.start()

        def get_result(self):
            return self.result


if __name__ == '__main__':
    # Select Paths
    face_path = './datasets/experement/old1.jpg'
    model_path = './trained_models/trained_model.pkl'

    # Run Analysing
    aging_diag = AgingDiagnosis()
    aging_diag.open_model(model_path)                            # Activate the trained model
    result = aging_diag.analyse_factors(face_path)               # Analyse factors

    # Show Results
    if result != None:
        age = result['Age']
        wrinkle_value = result['Wrinkle Features']
        print('\nAppearance Age:', age)
        print('Wrinkles: {:.2f}%'.format(wrinkle_value * 100.0))

        # Show Image
        img = cv2.imread(face_path)
        img = imutils.resize(img, width=500)
        cv2.imshow("Face Image", img)
        cv2.waitKey(0)
    else:
        print('There is ERROR')

