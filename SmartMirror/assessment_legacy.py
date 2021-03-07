import sys

import cv2
from keras.models import load_model
import numpy as np

from datasets import get_labels
from preprocessor import preprocess_input

from pathlib import Path
from keras.utils.data_utils import get_file
from wide_resnet import WideResNet

from MH.Autonomous_System_in_SMS.face_detections import FaceDetector, FaceRecognizer


class EmotionDetector:
    # parameters for loading data and images
    def __init__(self):
        self.emotion_model_path = '.\\models\\fer2013_mini_XCEPTION.70-0.66.hdf5'
        self.emotion_labels = get_labels('fer2013')

        # hyper-parameters for bounding boxes shape
        self.frame_window = 10
        self.emotion_offsets = (20, 40)

        # loading models
        self.emotion_classifier = load_model(self.emotion_model_path, compile=False)

        # getting input model shapes for inference
        self.emotion_target_size = self.emotion_classifier.input_shape[1:3]

        # starting lists for calculating modes
        self.emotion_window = []

    def detect(self, face_frame):
        try:
            gray_face = cv2.cvtColor(face_frame, cv2.COLOR_BGR2GRAY)
            gray_face = cv2.resize(gray_face, (self.emotion_target_size))
        except:
            return ''
        gray_face = preprocess_input(gray_face, True)
        gray_face = np.e

        expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = self.emotion_classifier.predict(gray_face)
        emotion_label_arg = np.argmax(emotion_prediction)

        return emotion_label_arg


class AgeDetector:
    def __init__(self):
        weight_file = None

        pretrained_model = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.5/weights.28-3.73.hdf5"
        modhash = 'fbe63257a054c1c5466cfd7bf14646d6'

        if not weight_file:
            weight_file = get_file("weights.28-3.73.hdf5", pretrained_model, cache_subdir="models",
                                   file_hash=modhash, cache_dir=str(Path(__file__).resolve().parent))

        # load model and weights
        self.img_size = 64
        self.model = WideResNet(self.img_size, depth=16, k=8)()
        self.model.load_weights(weight_file)

    def detect(self, face_frame):
        faces = np.empty((1, self.img_size, self.img_size, 3))

        faces[0, :, :, :] = cv2.resize(face_frame, (self.img_size, self.img_size))

        # predict ages and genders of the detected faces
        results = self.model.predict(faces)
        predicted_genders = results[0]
        ages = np.arange(0, 101).reshape(101, 1)
        predicted_ages = results[1].dot(ages).flatten()

        return int(predicted_ages[0])


if __name__ == '__main__':
    ed = EmotionDetector()
    ad = AgeDetector()
    cap = cv2.VideoCapture(0)
    fd = FaceDetector()
    fr = FaceRecognizer()
    while True:
        _, frame = cap.read()
        fd.detect(frame)
        frame, face = fd.choose_face(frame)
        if face is not None:
            #print(ad.detect(face))
            print(ed.detect(face))
            cv2.imshow("asdf", face)
        # print(fr.recognize(face))
        cv2.imshow("Smart Mirror System", frame)  # display image
        key = cv2.waitKey(1)
        if key == ord('q'):
            # To stop threads
            break