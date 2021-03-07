"""
File: train_emotion_classifierO.py
Date: 2019. 07. 11
Author: MK
Description: Train emotion classification model using logistic regression, Decision tree, SVM, KNN and Naive Bayes
Genearate the trained model for prediction.
"""
import os
from time import time

import dlib
import cv2
import imutils
from imutils import face_utils

from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from emotion_factor_dataframe import EmotionFactorAnalyzer


class FactorTraining:
    def __init__(self):
        self.feature_cols = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11']
        self.emotion_distance = pd.read_csv("emotion_distance.csv")  # load dataset
        self.emotion_distance = self.emotion_distance.sample(frac=1)  # shuffle the data
        self.X = self.emotion_distance[self.feature_cols]  # Features
        self.y = self.emotion_distance.emotion  # Target variable
        # print(self.X)
        # print(self.y)
        # print(self.emotion_distance.head())
        # emotion_distance.info()
        # emotion_distance.describe()

    def prepare_dt(self):
        # split X and y into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.25, random_state=0)
        # print("Testx", X_train)
        # print("Testy", y_train)
        # print("Testx", X_test)
        # print("Testy", y_test)

        return X_train, X_test, y_train, y_test

    def train_logreg_model(self):
        X_train, X_test, y_train, y_test = self.prepare_dt()
        logmodel = LogisticRegression(C=1, random_state=42)  # instantiate the model (using the default parameters)
        logmodel.fit(X_train, y_train)

        joblib.dump(logmodel, 'models/logmodel_model.pkl')  # Save the model as a pickle in a file
        logmodel_from_joblib = joblib.load('models/logmodel_model.pkl')  # Load the model from the file
        logmodel_predictions = logmodel_from_joblib.predict(X_test)  # Use the loaded model to make predictions
        # print(logmodel_predictions)

        # creating a confusion matrix
        cm = confusion_matrix(y_test, logmodel_predictions)
        print('Confusion matrix\n', cm)

        # model accuracy for X_test
        accuracy = logmodel_from_joblib.score(X_test, y_test)
        print('Accuracy of saved model  = ', accuracy)
        # Check that the loaded model is the same as the original
        assert logmodel.score(X_test, y_test) == logmodel_from_joblib.score(X_test, y_test)

        return

    def train_dtree_model(self):
        X_train, X_test, y_train, y_test = self.prepare_dt()
        # training a DescisionTreeClassifier
        dtree_model = DecisionTreeClassifier(max_depth=2, random_state=42).fit(X_train, y_train)

        joblib.dump(dtree_model, 'models/dtree_model.pkl')  # Save the model as a pickle in a file
        dtree_from_joblib = joblib.load('models/dtree_model.pkl')  # Load the model from the file
        dtree_predictions = dtree_from_joblib.predict(X_test)  # Use the loaded model to make predictions
        # print(dtree_predictions)

        # creating a confusion matrix
        cm = confusion_matrix(y_test, dtree_predictions)
        print('Confusion matrix\n', cm)

        # model accuracy for X_test
        accuracy = dtree_from_joblib.score(X_test, y_test)
        print('Accuracy of saved model = ', accuracy)
        # Check that the loaded model is the same as the original
        assert dtree_model.score(X_test, y_test) == dtree_from_joblib.score(X_test, y_test)

        # cross-validation
        depth = []
        for i in range(2, 20):
            clf = DecisionTreeClassifier(max_depth=i, random_state=42)
            # Perform 5-fold cross validation
            cv_scores = cross_val_score(estimator=clf, X=self.X, y=self.y, cv=5, n_jobs=4)
            depth.append((i, cv_scores.mean()))
        print('Decision Tree Cross validation\n', cv_scores)
        print("SVM CV mean accuracy: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() * 2))
        print('Decision Tree Cross validation\n', depth)

        return

    def train_svc_model(self):
        X_train, X_test, y_train, y_test = self.prepare_dt()
        # print('test value\n', X_test)
        t0 = time()
        svm_model_linear = SVC(kernel='linear', C=1, random_state=42).fit(X_train, y_train)  # random parameter
        print('training time: ', round(time() - t0, 3), 's')

        joblib.dump(svm_model_linear, 'models/svm_model.pkl')  # Save the model as a pickle in a file
        svm_from_joblib = joblib.load('models/svm_model.pkl')  # Load the model from the file
        svm_predictions = svm_from_joblib.predict(X_test)  # Use the loaded model to make predictions
        # print(svm_predictions)

        # creating a confusion matrix
        cm = confusion_matrix(y_test, svm_predictions)
        print('Confusion matrix\n', cm)

        # model accuracy for X_test
        accuracy = svm_from_joblib.score(X_test, y_test)
        print('Accuracy of saved model = ', accuracy)
        # Check that the loaded model is the same as the original
        assert svm_model_linear.score(X_test, y_test) == svm_from_joblib.score(X_test, y_test)

        # cross-validation
        clf = SVC(kernel='linear', C=1, random_state=42)
        cv_scores = cross_val_score(clf, self.X, self.y, cv=5)  # cv= number of time the score is computed

        # print each cv score (accuracy) and average them
        print('SVM Cross validation\n', cv_scores)
        print("SVM CV mean accuracy: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() * 2))

        return

    def train_knn_model(self):
        X_train, X_test, y_train, y_test = self.prepare_dt()
        knn = KNeighborsClassifier(n_neighbors=12).fit(X_train, y_train)  # random parameter

        joblib.dump(knn, 'models/knn_model.pkl')  # Save the model as a pickle in a file
        knn_from_joblib = joblib.load('models/knn_model.pkl')  # Load the model from the file
        knn_predictions = knn_from_joblib.predict(X_test)  # Use the loaded model to make predictions
        # print(knn_predictions)

        # creating a confusion matrix
        cm = confusion_matrix(y_test, knn_predictions)
        print('Confusion matrix\n', cm)

        # model accuracy for X_test
        accuracy = knn_from_joblib.score(X_test, y_test)
        print('Accuracy = ', accuracy)
        # Check that the loaded model is the same as the original
        assert knn.score(X_test, y_test) == knn_from_joblib.score(X_test, y_test)

        # cross-validation
        knn_cv = KNeighborsClassifier(n_neighbors=13)
        cv_scores = cross_val_score(knn_cv, self.X, self.y, cv=5)  # cv= number of time the score is computed

        # print each cv score (accuracy) and average them
        print('Knn cross validation\n', cv_scores)
        print("KNN CV mean accuracy: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() * 2))  # mean accuracy
        # print('cv_scores mean: {}'.format(np.mean(cv_scores)))

        return

    def train_gnb_model(self):
        X_train, X_test, y_train, y_test = self.prepare_dt()
        # print(X_test)
        gnb = GaussianNB().fit(X_train, y_train)

        joblib.dump(gnb, 'models/gnb_model.pkl')  # Save the model as a pickle in a file
        gnb_from_joblib = joblib.load('models/gnb_model.pkl')  # Load the model from the file
        gnb_predictions = gnb_from_joblib.predict(X_test)  # Use the loaded model to make predictions
        # print(gnb_predictions)

        # creating a confusion matrix
        cm = confusion_matrix(y_test, gnb_predictions)
        print('Confusion matrix\n', cm)

        # model accuracy for X_test
        accuracy = gnb.score(X_test, y_test)
        print('Accuracy = ', accuracy)
        # Check that the loaded model is the same as the original
        assert gnb.score(X_test, y_test) == gnb_from_joblib.score(X_test, y_test)

        return

    def hypertune_parameters(self, model=''):
        if model == 'knn':
            # create new a knn model
            knn2 = KNeighborsClassifier()
            # create a dictionary of all values we want to test for n_neighbors
            param_grid = {'n_neighbors': np.arange(1, 31)}
            # use gridsearch to test all values for n_neighbors
            knn_gscv = GridSearchCV(knn2, param_grid, cv=5)
            # fit model to data
            knn_gscv.fit(self.X, self.y)

            # check top performing n_neighbors value
            print('Knn best parameters=', knn_gscv.best_params_)
            # check mean score for the top performing value of n_neighbors
            print('Knn accuracy with optimal parameters=', knn_gscv.best_score_)

        elif model == 'svm':
            X_train, X_test, y_train, y_test = self.prepare_dt()
            # create a dictionaries of all values we want to test by cross-validation
            tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                                 'C': [1, 10, 100, 1000]},
                                {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
            scores = ['precision', 'recall']

            for score in scores:
                print("# Tuning hyper-parameters for %s" % score)
                print()

                # create new a svm model
                clf = GridSearchCV(SVC(), tuned_parameters, cv=5, scoring='%s_macro' % score)
                clf.fit(X_train, y_train)

                print("Best parameters set found on development set:\n")
                print(clf.best_params_)
                print("\nGrid scores on development set:\n")
                means = clf.cv_results_['mean_test_score']
                stds = clf.cv_results_['std_test_score']
                for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

                print("\nDetailed classification report:\n")
                print("The model is trained on the full development set.\n")
                print("The scores are computed on the full evaluation set.")
                y_true, y_pred = y_test, clf.predict(X_test)
                print(classification_report(y_true, y_pred))


class FactorDetector:
    def __init__(self):
        # Load the model from the files
        self.svm_model = joblib.load('.\\models\\svm_model.pkl')
        self.log_model_path = joblib.load('.\\models\\logmodel_model.pkl')
        self.gnb_model_path = joblib.load('.\\models\\gnb_model.pkl')

        self.efa = EmotionFactorAnalyzer()  # emotion factor analyzer object
        self.detector = dlib.get_frontal_face_detector()
        self.predictor_path = "shape_predictor_68_face_landmarks.dat"
        self.predictor = dlib.shape_predictor(self.predictor_path)

    def detect_emotion_factor(self, frame):
        # distance_vector = self.efa.preprocessing(image)

        global distance_vector, face_shape, result, model_name
        emotions = ['Angry', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        models = {'SVC': self.svm_model, 'Logistic Regression': self.log_model_path, 'Naive Bayes': self.gnb_model_path}

        # # load the input image
        # emotion = os.path.dirname(frame)  # get the name of the folder as emotion
        # image = cv2.imread(frame)
        # shape = image.shape
        emotion = ' '
        shape = frame.shape
        # print(shape[:])
        # print('emotion = ', emotion)
        # cv2.imshow("Original image", image)
        # cv2.waitKey(1)

        while shape:
            # if shape[0] < 300 or shape[1] < 300:  # check image size
            #     print('image size incorrect')
            #     break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = self.detector(gray, 1)  # detect faces in the grayscale image
            if len(rects) != 1:  # check only one person
                print('Face no detected')
                break

            target_face = rects[0]
            height = target_face.height()
            width = target_face.width()
            if height < 150 or width < 150:  # check face part size to detect
                print('Bad size')
                break

            # Crop face part include hear and neck
            img_cropped = frame[target_face.top() - 50:target_face.bottom() + 80,
                          target_face.left() - 50:target_face.right() + 50]
            # cv2.imshow("Cropped image", img_cropped)
            # cv2.waitKey(1)

            img_resize = imutils.resize(img_cropped, width=500)
            # cv2.imshow("Resized image", img_resize)
            # cv2.waitKey(1)

            gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
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
                cv2.rectangle(img_resize, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # show the face number
                cv2.putText(img_resize, "Face #{}".format(i + 1), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # loop over the (x, y)-coordinates for the facial landmarks and draw them on the image
                k = 0
                for (x, y) in face_shape[:]:
                    cv2.circle(img_resize, (x, y), 1, (0, 0, 255), -1)
                    cv2.putText(img_resize, str(k), (x, y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.3, (0, 255, 0))
                    k += 1

                # cv2.imwrite(save_path, gray)  # Save cropped gray image

            result, distance_vector = self.efa.compute_factors(face_shape, emotion)
            # print('Distance result ', result, type(result))

            print('f1:The distance between the right eye and the right eyebrow:', result['f1'])
            print('f2:The distance between the left eye and the left eyebrow:', result['f2'])
            print('f3:The distance between the inner ends of the eyebrows:', result['f3'])
            print('f4:The distance between the outer corner of the right eye and the corner of the mouth:',
                  result['f4'])
            print('f5:The distance between the outer corner of the left eye and the corner of the mouth:', result['f5'])
            print('f6:The distance of the right eyelid from the right eyebrow:', result['f6'])
            print('f7:The distance of the left eyelid from the left eyebrow:', result['f7'])
            print('f8: The distance of the right upper and lower eyelids:', result['f8'])
            print('f9: The distance of the left upper and lower eyelids:', result['f9'])
            print('f10: The distance between the upper and lower lip:', result['f10'])
            print('f11: The distance between the outer ends of the mouth:', result['f11'])

            # print('Distance vector ', distance_vector, type(distance_vector))
            del distance_vector['emotion']
            # print('Distance vector 2', distance_vector, type(distance_vector))

            df = pd.DataFrame(distance_vector)
            # print('Data frame\n', df)
            # print('Data frame values\n', df.values)

            # for model_name, val in models.items():
            # emotion_prediction = val.predict(df.values)  # Use the loaded model to make predictions
            emotion_prediction = models['SVC'].predict(df.values)  # Use the loaded model to make predictions
            value_predicted = int("".join(map(str, emotion_prediction)))

            for emotion in emotions:
                emotion_label = emotions[value_predicted]
            print('\nEmotion predicted\n Value: {} \n Emotion: {}'.format(value_predicted, emotion_label))

            # show the output image with the face detections + facial landmarks
            # cv2.imshow("Resized image+Landmarks", img_resize)
            # cv2.waitKey(0)

            break

        return distance_vector, emotion_label

    def capture_img(self):
        # cap = cv2.VideoCapture(0)
        #
        # while True:
        #     _, frame = cap.read()
        #     rect = fd.detect(frame)
        #
        #     cv2.imshow("Face", frame)
        #
        #     key_input = cv2.waitKey(1)
        #     if key_input & 0xFF == ord('q'):
        #         break

        cam = cv2.VideoCapture(0)
        cv2.namedWindow("Smart Mirror")

        while True:
            ret, frame = cam.read()
            cv2.imshow("Smart Mirror", frame)
            if not ret:
                break
            k = cv2.waitKey(1) & 0xFF

            if k == 27:  # ESS exit the program
                print("Escape hit, closing...")
                break
            result, emotion = factor_detector.detect_emotion_factor(frame)
            print('Distance computed\n{} \nEmotion: {}'.format(result, emotion))


if __name__ == '__main__':
    # ft = FactorTraining()
    # print('********************* LOGREG *******************************\n')
    # ft.train_logreg_model()
    # print('\n********************* Decision Tree *******************************')
    # ft.train_dtree_model()
    # print('\n********************* SVM *******************************')
    # ft.train_svc_model()
    # # print('\n********************* SVM hypertune_parameters *******************************')
    # # ft.hypertune_parameters(model='svm')
    # print('\n********************* KNN *******************************')
    # ft.train_knn_model()
    # # print('\n********************* KNN hypertune_parameters *******************************')
    # # ft.hypertune_parameters(model='knn')
    # print('\n********************* Naive Bayes *******************************')
    # ft.train_gnb_model()

    # img = 'test/n.jpg'
    factor_detector = FactorDetector()
    factor_detector.capture_img()
    # result, emotion = factor_detector.detect_emotion_factor(img)
    # print('Distance computed\n{} \nEmotion: {}'.format(result, emotion))

