"""
File: emotion_factor_dataframe.py
Date: 2019. 07. 11
Author: MK
Description: Train emotion classification model using logistic regression
"""

import math
import os
import pandas as pd
from imutils import face_utils
import imutils
import dlib
import cv2
import numpy as np
import matplotlib.pyplot as plt


class EmotionFactorAnalyzer:
    def __init__(self):
        self.data_dir = "Big_dataset"

        # initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor_path = "./models/shape_predictor_81_face_landmarks.dat"
        self.predictor = dlib.shape_predictor(self.predictor_path)

        # distance vector dictionary initialisation
        self.distance_vector = {"f1": [],
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
                                'emotion': []}

    @staticmethod
    def compute_distance(p1, p2):
        distance = math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))
        return distance

    @staticmethod
    def compute_middle(p1, p2):
        p3 = (p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5
        return p3

    def compute_factors(self, shape, emotion='none'):
        width = self.compute_distance(shape[15], shape[1])
        height = self.compute_distance(shape[8], (shape[1][0], min(shape[72][1], shape[69][1])))

        # print('computing distance ...')
        result = {"f1": self.compute_distance(shape[19], self.compute_middle(shape[36], shape[39]))/height,
                  "f2": self.compute_distance(shape[24], self.compute_middle(shape[42], shape[45]))/height,
                  "f3": self.compute_distance(shape[21], shape[22])/width,
                  "f4": self.compute_distance(shape[36], shape[48])/height,
                  "f5": self.compute_distance(shape[45], shape[54])/height,
                  "f6": self.compute_distance(shape[19], shape[37])/height,
                  "f7": self.compute_distance(shape[24], shape[44])/height,
                  "f8": self.compute_distance(shape[37], shape[41])/height,
                  "f9": self.compute_distance(shape[44], shape[46])/height,
                  "f10": self.compute_distance(shape[62], shape[66])/height,
                  "f11": self.compute_distance(shape[48], shape[54])/width}
        # print('result\n', result)

        self.distance_vector['f1'].append(result['f1'])
        self.distance_vector['f2'].append(result['f2'])
        self.distance_vector['f3'].append(result['f3'])
        self.distance_vector['f4'].append(result['f4'])
        self.distance_vector['f5'].append(result['f5'])
        self.distance_vector['f6'].append(result['f6'])
        self.distance_vector['f7'].append(result['f7'])
        self.distance_vector['f8'].append(result['f8'])
        self.distance_vector['f9'].append(result['f9'])
        self.distance_vector['f10'].append(result['f10'])
        self.distance_vector['f11'].append(result['f11'])
        self.distance_vector['emotion'].append(emotion)

        return result, self.distance_vector

    def preprocessing(self):
        for root, dirs, files in os.walk(self.data_dir):
            # print(dirs)
            for filename in files:
                save_path = os.path.join(root, filename)
                print(save_path)
                emotion = os.path.basename(os.path.dirname(save_path))  # get the name of the folder as emotion

                # load the input image, crop it, resize it, and convert it to grayscale
                img = cv2.imread(os.path.join(root, filename))
                shape = img.shape
                print(shape[:])

                # if shape[0] < 300 or shape[1] < 300:        # check image size
                #     print('image size incorrect')
                #     continue

                # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # rects = self.detector(gray, 1)  # detect faces in the grayscale image
                # if len(rects) != 1:  # check only one person
                #     print('face no detected')
                #     continue

                # target_face = rects[0]
                # height = target_face.height()
                # width = target_face.width()
                # if height < 150 or width < 150:  # check face part size to detect
                #     print('Bad size')
                #     continue

                # # Crop face part include hear and neck
                # img_cropped = img[target_face.top() - 50:target_face.bottom() + 80,
                #       target_face.left() - 50:target_face.right() + 50]
                # # cv2.imshow("Cropped image", img_cropped)
                # # cv2.waitKey(1)
                #
                # img = imutils.resize(img, width=500)
                # # cv2.imshow("Resized image", img)
                # # cv2.waitKey(0)

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                rects = self.detector(gray, 1)  # detect faces in new cropped, resize, grayscale image

                # loop over the face detected
                for (i, rect) in enumerate(rects[0:1]):
                    # determine the facial landmarks for the face region, then convert the facial landmark
                    # (x, y)-coordinates to a NumPy array
                    face_shape = self.predictor(gray, rect)
                    face_shape = face_utils.shape_to_np(face_shape)
                    # print(shape)

                    # convert dlib's rectangle to a OpenCV-style bounding box
                    # [i.e., (x, y, w, h)], then draw the face bounding box
                    (x, y, w, h) = face_utils.rect_to_bb(rect)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # show the face number
                    cv2.putText(img, "Face #{}".format(i + 1), (x - 10, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # loop over the (x, y)-coordinates for the facial landmarks and draw them on the image
                    k = 0
                    for (x, y) in face_shape[:]:
                        cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
                        cv2.putText(img, str(k), (x, y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.3, (0, 255, 0))
                        k += 1

                    # show the output image with the face detections + facial landmarks
                    # cv2.imshow("Resized image+Landmarks", img)
                    # cv2.waitKey(0)

                    # cv2.imwrite(save_path, gray)  # Save cropped gray image

                result, distance_vector = self.compute_factors(face_shape, emotion)
                # print('Distance result ', result, type(result))
                # print('Distance vector ', distance_vector, type(distance_vector))

        print(self.distance_vector)  # final distance vector

    def make_dataframe(self):
        """
        Convert the dictionary into DataFrame
        :return: pandas dataframe
        """
        # print(self.distance_vector)
        print('\n>>>>>>>>>>Dataframe in progress')
        return pd.DataFrame(self.distance_vector)

    @staticmethod
    def write_csv_file(frame_df, path="emotion_distance.csv"):
        print('\n>>>>>>>>>>CSV writing in progress')
        if os.path.exists('emotion_distance.csv'):
            os.remove('emotion_distance.csv')  # this deletes the file
            print("Existing  csv file has been overwritten")  # add this to prevent errors
            frame_df.to_csv(path, mode='a', index=False)
        else:
            frame_df.to_csv(path, mode='a', index=False)

    @staticmethod
    def compute_corr():
        data = pd.read_csv("emotion_distance.csv")  # load dataset

        corr = data.corr()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
        fig.colorbar(cax)
        ticks = np.arange(0, len(data.columns), 1)
        ax.set_xticks(ticks)
        plt.xticks(rotation=90)
        ax.set_yticks(ticks)
        ax.set_xticklabels(data.columns)
        ax.set_yticklabels(data.columns)
        plt.title('Emotion Landmarks Distance Factor Correlation')
        plt.show()

        print(data.corr())
        print(data.corr()["emotion"].sort_values(ascending=False))


if __name__ == '__main__':
    efa = EmotionFactorAnalyzer()
    efa.preprocessing()
    df = efa.make_dataframe()
    print(df)
    efa.write_csv_file(df)
    efa.compute_corr()


