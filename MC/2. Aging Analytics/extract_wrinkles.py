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


"""
    2. Extracting Features
"""

predictor_path = 'shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

data_dir = "../datasets/wrinkles4"
file_list = os.listdir(data_dir)
file_list.sort()
count = 0
frame = {"Age": [],
         "Density_Corner_Left_Eye": [],
         "Density_Corner_Right_Eye": [],
         "Density_Forehead": [],
         "Density_Cheek_Left": [],
         "Density_Cheek_Right": [],

         "Depth_Corner_Left_Eye": [],
         "Depth_Corner_Right_Eye": [],
         "Depth_Forehead": [],
         "Depth_Cheek_Left": [],
         "Depth_Cheek_Right": [],

         "Variance_Corner_Left_Eye": [],
         "Variance_Corner_Right_Eye": [],
         "Variance_Forehead": [],
         "Variance_Cheek_Left": [],
         "Variance_Cheek_Right": [],
         "Path": []}
age = []
path = []
threshold = 40
start_time = time.time()

for i in file_list:
    try:
        image = cv2.imread(data_dir + "/" + i)
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
    except:
        continue

    # Skip NaN
    if math.isnan(left_eye_wr) or math.isnan(right_eye_wr) or math.isnan(forehead_wr) or\
            math.isnan(cheek_left_wr) or math.isnan(cheek_right_wr):
        continue

    if math.isnan(d1) or math.isnan(d2) or math.isnan(d3) or math.isnan(d4) or math.isnan(d5):
        continue

    if math.isnan(v1) or math.isnan(v2) or math.isnan(v3) or math.isnan(v4) or math.isnan(v5):
        continue

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

    # Read Age and Path from "2_1937-09-06_2010.jpg"
    count += 1
    f = i.split("_")
    birth_year = int(f[1].split("-")[0])
    taken_year = int(f[2].split('.')[0])
    a = (taken_year - birth_year)
    age.append(a)
    path.append(i)

    # Wrinkles
    depth = [d1, d2, d3, d4, d5]
    variance = [v1, v2, v3, v4, v5]
    density = [left_eye_wr, right_eye_wr, forehead_wr, cheek_left_wr, cheek_right_wr]

    # Show Results
    print('Age:', taken_year - birth_year)
    print('Path:', i)
    print('Wrinkle Density', density)
    print('Wrinkle Depth', depth)
    print('Wrinkle Variance', variance)

    # Save Results in Frame
    frame['Age'].append(a)
    frame["Path"].append(data_dir + "/" + i)

    frame['Density_Corner_Left_Eye'].append(left_eye_wr)
    frame["Density_Corner_Right_Eye"].append(right_eye_wr)
    frame['Density_Forehead'].append(forehead_wr)
    frame["Density_Cheek_Left"].append(cheek_left_wr)
    frame['Density_Cheek_Right'].append(cheek_right_wr)

    frame['Depth_Corner_Left_Eye'].append(d1)
    frame["Depth_Corner_Right_Eye"].append(d2)
    frame['Depth_Forehead'].append(d3)
    frame["Depth_Cheek_Left"].append(d4)
    frame['Depth_Cheek_Right'].append(d5)

    frame['Variance_Corner_Left_Eye'].append(v1)
    frame["Variance_Corner_Right_Eye"].append(v2)
    frame['Variance_Forehead'].append(v3)
    frame["Variance_Cheek_Left"].append(v4)
    frame['Variance_Cheek_Right'].append(v5)

# Save Results in CSV File
# df = pd.DataFrame(frame)
# df.to_csv('./data/data7.csv', mode='a')
# print('\nCount of Images:', count)
# print("--- %s seconds ---" % (time.time() - start_time))

# Count of Images: 11295
# --- 1636.283148765564 seconds ---


