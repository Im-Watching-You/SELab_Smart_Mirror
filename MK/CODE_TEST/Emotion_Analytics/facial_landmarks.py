# import the necessary packages
# from 5_facial_landmarks.imutilsmaster.imutils import face_utils
import glob
import math
import os
import pandas as pd
from imutils import face_utils
import imutils
import dlib
import cv2
import numpy as np
import argparse


def compute_distance(p1, p2):
    distance = math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))
    return distance


def midpoint(pt1, pt2):
    return (pt1[0] + pt2[0]) * 0.5, (pt1[1] + pt2[1]) * 0.5


predictor_path = 'shape_predictor_81_face_landmarks.dat'
faces_folder_path = 'test'

# distance vector dictionary initialisation
distance_vector = {'f1': [],
                   'f2': [],
                   'f3': [],
                   'f4': [],
                   'f5': [],
                   'f6': [],
                   'f7': [],
                   'f8': [],
                   'f9': [],
                   'f10': [],
                   'f11': [],
                   'emotion': []
                   }

# initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

for root, dirs, files in os.walk(faces_folder_path):
    print(dirs)
    for filename in files:
        if '.jpg' in filename:
            print(os.path.join(root, filename))

# for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
            # load the input image, resize it, and convert it to grayscale
            image = cv2.imread(os.path.join(root, filename))
            image = imutils.resize(image, width=500)

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # detect faces in the grayscale image
            rects = detector(gray, 1)

            # loop over the face detections
            for (i, rect) in enumerate(rects):
                # determine the facial landmarks for the face region, then convert the facial landmark
                # (x, y)-coordinates to a NumPy array
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                # print(shape)
                #
                # Compute the distance for and add the result into the list factor
                distance1 = compute_distance(shape[19], midpoint(shape[36], shape[39]))
                distance_vector['f1'].append(distance1)
                distance2 = compute_distance(shape[24], midpoint(shape[42], shape[45]))
                distance_vector['f2'].append(distance2)
                distance3 = compute_distance(shape[21], shape[22])
                distance_vector['f3'].append(distance3)
                distance4 = compute_distance(shape[38], shape[48])
                distance_vector['f4'].append(distance4)
                distance5 = compute_distance(shape[45], shape[54])
                distance_vector['f5'].append(distance5)
                distance6 = compute_distance(shape[19], shape[37])
                distance_vector['f6'].append(distance6)
                distance7 = compute_distance(shape[24], shape[44])
                distance_vector['f7'].append(distance7)
                distance8 = compute_distance(shape[37], shape[41])
                distance_vector['f8'].append(distance8)
                distance9 = compute_distance(shape[44], shape[46])
                distance_vector['f9'].append(distance9)
                distance10 = compute_distance(shape[62], shape[66])
                distance_vector['f10'].append(distance10)
                distance11 = compute_distance(shape[48], shape[54])
                distance_vector['f11'].append(distance11)
                distance_vector['emotion'].append(distance11)

                #
                # print('f1:The distance between the right eye and the right eyebrow:', distance1)
                # print('f2:The distance between the left eye and the left eyebrow:', distance2)
                # print('f3:The distance between the inner ends of the eyebrows:', distance3)
                # print('f4:The distance between the outer corner of the right eye and the corner of the mouth:', distance4)
                # print('f5:The distance between the outer corner of the left eye and the corner of the mouth:', distance5)
                # print('f6:The distance of the right eyelid from the right eyebrow:', distance6)
                # print('f7:The distance of the left eyelid from the left eyebrow:', distance7)
                # print('f8: The distance of the right upper and lower eyelids:', distance8)
                # print('f9: The distance of the left upper and lower eyelids:', distance9)
                # print('f10: The distance between the upper and lower lip:', distance10)
                # print('f11: The distance between the outer ends of the mouth:', distance11)
                #

                # convert dlib's rectangle to a OpenCV-style bounding box
                # [i.e., (x, y, w, h)], then draw the face bounding box
                (x, y, w, h) = face_utils.rect_to_bb(rect)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # show the face number
                cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # loop over the (x, y)-coordinates for the facial landmarks and draw them on the image
                k = 0
                for (x, y) in shape[:]:
                    cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
                    cv2.putText(image, str(k), (x, y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.3, (0, 255, 0))
                    k += 1

            # show the output image with the face detections + facial landmarks
            cv2.imshow("Output", image)
            cv2.waitKey(0)

print(distance_vector, type(distance_vector))

# Convert the dictionary into DataFrame
distance_vector_pd = pd.DataFrame(distance_vector)
print(distance_vector_pd)

# Generate the csv dataset file
# distance_vector_pd.to_csv('emotion_distance.csv', index=False)
