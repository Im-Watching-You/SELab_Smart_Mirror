"""
Date: 2019.07.08
Programmer: MK
Description: class managing the analytics of the emotions.

This file contains the functionaries of the module Emotion analytics

"""


class EmotionPrediction:
    def __init__(self):
        self.dataset_path = ""

    def prepare_dataset(self):
        """
        Evaluate the number of pictures in the dataset available for the training
        and create a training set and testing set.
        """
        print(self.dataset_path)

    def preprocess_dataset(self, image):
        """
        This method preprocess the dataset for the training model through various operations
        (face alignment, data augmentation, face normalization).
        :return: ndarray
        """
        return

    def extract_features(self, image):
        """"
        Detect then extract the features from the passed input images and extract them for the training method.
        :return dictionary
        """
        extracted_feat = {'a', 'b'}
        return extracted_feat

    def train_model(self, features_vector):
        """"
        Train the machine learning model by fitting the training set data.
        :return
        """
        extracted_feat = {'a', 'b'}
        return extracted_feat

    def estimate_model(self, image):
        """
        The method estimates the accuracy of the trained model by proceeding a prediction with the test dataset.
        :return float
        """
        model_accuracy = 0.75
        return model_accuracy

    def predict_emotion(self, image):
        """"
        Main method of the class Emotion Analyzer. Get the frame as an input value,
        predict the emotion by extracting the features and classifying them.
        Return the label corresponding to the emotion index predicted
        :return string label of the detected emotion
        """
        emotion_lab = 'Happiness'
        return emotion_lab

    def save_emotion(self, user_id, img_path, emotion_label):
        """
        This method saves the predicted emotion into the database along with the path
        of the image on which the prediction has been done.
        This is to facilitate further analysis of emotion.
        :return:
        """


class FactorAnalyzer:
    def __init__(self):
        self.dataset_path = ""

    def prepare_dataset(self):
        """
        Evaluate the number of pictures in the dataset available for the training and
        create a training set and testing set.
        """
        print(self.dataset_path)

    def preprocess_dataset(self, image):
        """
        This method preprocesses the dataset for the training model through various
        operations (face alignment, data augmentation, face normalization).
        :return: ndarray
        """
        return

    def detect_face_lm(image):
        """
        This method detects the face from the inputted picture, then detect the landmarks points from the face.
        :return:
        """
        lm_points = []
        return lm_points

    def compute_lmdistance(self):
        """
        Extract and compute the distance relation between face component landmarks and
        correlate the result with the emotion predicted by the prediction module to get
        the classification. (or can make its own prediction)
        :return:
        """
        lm_distance = []
        return lm_distance

    def extract_features(self, image):
        """"
        Extract the landmarks points from the passed input images for the training method.
        :return dictionary
        """
        extracted_feat = {'a', 'b'}
        return extracted_feat

    def train_model(self, features_vector):
        """"
        Train the machine learning model by fitting the landmarks points vector.
        :return
        """
        extracted_feat = {'a', 'b'}
        return extracted_feat

    def estimate_model(self, image):
        """
        The method estimates the accuracy of the trained model by proceeding a prediction with the test dataset.
        :return float
        """
        model_accuracy = 0.75
        return model_accuracy

    def predict_emotion(self, image):
        """"
        Classification of the extracted landmarks points to the matching emotion.
        Return the label corresponding to the emotion index predicted.
        :return string label of the detected emotion
        """
        emotion_lab = 'Happiness'
        return emotion_lab

    def display_factor(self, features_vector, emotion_label):
        """
        From the emotion landmarks point, generate a representation (graphic or AUs) of the
        factors used to predict the emotion.
        :return list of the factor and theirs values
        """
        emotion_factor = []
        return emotion_factor

    def save_factor(self, user_id, img_path, emotion_label, lmdistance):
        """
        This method saves the predicted landmark into the database along with the path
        of the image on which the prediction has been done.
        This is to facilitate further analysis of emotion.
        """


class ChangesAnalyzer:
    def __init__(self):
        self.dataset_path = ""

    def display_trends(self, user_id):
        """
        Display the emotion changes according to the trends
        :return:
        """

    def display_seasonable(self, user_id):
        """
        Display the emotion changes according to the seasons
        :return:
        """