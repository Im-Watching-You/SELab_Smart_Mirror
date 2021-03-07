import os
import threading

import cv2
import dlib
from imutils import face_utils
import pandas as pd
import numpy as np
import math
from sklearn.externals import joblib
from threading import Thread

class AgingGeoDetector(threading.Thread):

    def __init__(self, q, img):
        threading.Thread.__init__(self)
        self.detector = dlib.get_frontal_face_detector()
        p = "./models/shape_predictor_81_face_landmarks.dat"
        self.predictor = dlib.shape_predictor(p)
        self.age_predicter = joblib.load("./models/aging_factor/SVC.pkl")
        self.result = None
        self.queue = q
        self.img = img

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

    def extract_face_landmark(self, img):
        d = self.detector(img, 0)
        shape = self.predictor(img, d[0])  # find facial landmarks
        return shape

    def compute_age_factors(self):
        try:
            shape = self.extract_face_landmark(self.img)
            mtx_landmarks = np.matrix([[p.x, p.y] for p in shape.parts()])
            landmarks = np.squeeze(np.asarray(mtx_landmarks))
            width = self.compute_distance(landmarks[15], landmarks[1])
            height = self.compute_distance(landmarks[8], (landmarks[1][0], min(landmarks[72][1], landmarks[69][1])))
            r = self.compute_factors(landmarks, width, height)
            # Add Path
            # r['path'] =
            # Age detection code
            features = np.reshape(list(r.values()), (1, -1))
            r['age'] = self.age_predicter.predict(features)[0]

            self.result = {"Age": r['age'], "Factor":r}
        except:
            self.result = None
        return self.result

    def run(self):
        self.queue.put(self.compute_age_factors())

    def get_result(self):
        return self.result

    def train(self):
        pass

    def predict_age(self, r):

        return 0


if __name__ == '__main__':
    fp = AgingGeoDetector()
    # fp.run()
    # fp.renaming()
    img = cv2.imread("./resultss/27_1916-07-27_1950.jpg")
    shape=fp.extract_face_landmark(img)
    result = fp.compute_age_factors(shape)
    print(result)