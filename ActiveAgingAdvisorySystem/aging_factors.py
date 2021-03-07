"""
Date: 2019.07.05
Programmer: Maksym Chernozhukov
Description: Code for Aging Analytics
"""
import threading
from pathlib import Path
from wide_resnet import WideResNet
from keras.utils.data_utils import get_file
from ActiveAgingAdvisorySystem.assesment import AgingDiagnosis as DBAgingDiagnosis
import numpy as np
import cv2


class AgingAppearance:
    """
    Predicting Age
    """
    def __init__(self):
        weight_file = None

        pretrained_model = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.5/weights.28-3.73.hdf5"
        modhash = 'fbe63257a054c1c5466cfd7bf14646d6'

        if not weight_file:
            weight_file = get_file("weights.28-3.73.hdf5", pretrained_model, cache_subdir="models",
                                   file_hash=modhash, cache_dir=str(Path(__file__).resolve().parent))

        # load model and weights
        self.db_ad = DBAgingDiagnosis()
        self.img_size = 64
        self.model_id = 1
        self.model = WideResNet(self.img_size, depth=16, k=8)()
        self.model.load_weights(weight_file)
        self.curr_ass_id = -1
        self.result = None


    def make_preprocessing(self, dataset):
        """
        To make preprocessing for initial data like cropping and aligning faces, and etc.
        :param dataset: array, with initial data.
        :return: array, with processed dataset.
        """
        result = []
        return result

    def seperate_dataset(self, processed_dataset):
        """
        To separate dataset on train_set and test_set.
        :param processed_dataset: array, with processed dataset.
        :return: train_set, test_set on which the model will be train and evaluate
        """
        result = []
        return result

    def train_model(self, train_set):
        """
        To train a model on train_set using machine learning algorithms.
        :param train_set: array, dataset on which the model will be train.
        :return: the trained model which can be used for prediction target's apparent age.
        """
        result = False
        return result

    def estimate_model(self, test_set):
        """
        To estimates the trained model on test_set to define accuracy of the model.
        :param test_set: array, dataset on which the model will be evaluate.
        :return: To estimates the trained model on test_set to define accuracy of the model.
        """
        result = 0.
        return result

    def predict_age(self, face_image, data):
        """
        To predict target’s apparent age by input image face data.
        :param face_image: ndarray, the matrix consists of pixels of the face image.
        :return: int, predicted target’s apparent age.
        result = 25
        """
        # def th(face_image, data):
            # try:
        faces = np.empty((1, self.img_size, self.img_size, 3))

        faces[0, :, :, :] = cv2.resize(face_image, (self.img_size, self.img_size))

        # predict ages and genders of the detected faces
        results = self.model.predict(faces)
        predicted_genders = results[0]
        ages = np.arange(0, 101).reshape(101, 1)
        predicted_ages = results[1].dot(ages).flatten()
        # print(">>>>> ", int(predicted_ages[0]))
        self.curr_ass_id = self.db_ad.register_assessment({'model_id': self.model_id, 'photo_id': data['photo_id'], "session_id": data["session_id"],
                                           "result": int(predicted_ages[0])})
        self.result = int(predicted_ages[0])
            # except:
            #     # self.result = None
            #     pass
        # threading.Thread(target=th, args=(face_image,data,)).start()

    def get_result(self):
        return self.result

    def get_curr_predicted_age(self, data):
        return self.db_ad.retrieve_latest_assessment_by_ids({'model_id': self.model_id, 'photo_id': data['photo_id'], "session_id": data["session_id"]})

class AgingDiagnosis:
    """
    Analyzing Factors for Age Appearance
    """
    def make_preprocessing(self, dataset):
        """
        To make preprocessing for initial data like cropping and aligning faces, and etc.
        :param dataset: array, with initial data.
        :return: array, with processed dataset.
        """
        result = []
        return result

    def seperate_dataset(self, processed_dataset):
        """
        To separate dataset on train_set and test_set.
        :param processed_dataset: array, with processed dataset.
        :return: train_set, test_set on which the model will be train and evaluate
        """
        result = []
        return result

    def train_model(self, train_set):
        """
        To train a model on train_set using machine learning algorithms.
        :param train_set: array, dataset on which the model will be train.
        :return: the trained model which can be used for prediction target's apparent age.
        """
        result = False
        return result

    def estimate_model(self, test_set):
        """
        To estimates the trained model on test_set to define accuracy of the model.
        :param test_set: array, dataset on which the model will be evaluate.
        :return: To estimates the trained model on test_set to define accuracy of the model.
        """
        result = 0.
        return result

    def analyse_factors(self, face_image):
        """
        To analyze the various factors which contribute to the determination of the age.
        :param face_image: ndarray, the matrix consists of pixels of the face image.
        :return: dict, containing the factors names with their values which used to predict the age of the user.
        result = {'Wrinkle Density': '0.87', 'Aging Spot': '0.5'}
        """
        result = {"Factor 1":30, "Factor 2": 40, "Factor 3": 20}
        return result


class AgingTrend:
    """
    Trends Aging Progress
    """
    def make_preprocessing(self, dataset):
        """
        To make preprocessing for initial data like cropping and aligning faces, and etc.
        :param dataset: array, with initial data.
        :return: array, with processed dataset.
        """
        result = []
        return result

    def seperate_dataset(self, processed_dataset):
        """
        To separate dataset on train_set and test_set.
        :param processed_dataset: array, with processed dataset.
        :return: train_set, test_set on which the model will be train and evaluate
        """
        result = []
        return result

    def train_model(self, train_set):
        """
        To train a model on train_set using machine learning algorithms.
        :param train_set: array, dataset on which the model will be train.
        :return: the trained model which can be used for prediction target's apparent age.
        """
        result = False
        return result

    def estimate_model(self, test_set):
        """
        To estimates the trained model on test_set to define accuracy of the model.
        :param test_set: array, dataset on which the model will be evaluate.
        :return: To estimates the trained model on test_set to define accuracy of the model.
        """
        result = 0.
        return result

    def analyse_trends_progress(self, aging_progress_rate):
        """
        To analyze the trends for the same person by comparing the past results on the aging progress.
        :param aging_progress_rate: dict, rate of aging progress over a period of time with description.
        :return: dict, kind of trends and how quickly it changes based over a specified period of time.
        result = {'Increase': 'slow'}
        """
        result = {}
        return result


class AgingSeasonableVariator:
    """
    Seasonable Variations Aging Progress
    """
    def make_preprocessing(self, dataset):
        """
        To make preprocessing for initial data like cropping and aligning faces, and etc.
        :param dataset: array, with initial data.
        :return: array, with processed dataset.
        """
        result = []
        return result

    def seperate_dataset(self, processed_dataset):
        """
        To separate dataset on train_set and test_set.
        :param processed_dataset: array, with processed dataset.
        :return: train_set, test_set on which the model will be train and evaluate
        """
        result = []
        return result

    def train_model(self, train_set):
        """
        To train a model on train_set using machine learning algorithms.
        :param train_set: array, dataset on which the model will be train.
        :return: the trained model which can be used for prediction target's apparent age.
        """
        result = False
        return result

    def estimate_model(self, test_set):
        """
        To estimates the trained model on test_set to define accuracy of the model.
        :param test_set: array, dataset on which the model will be evaluate.
        :return: To estimates the trained model on test_set to define accuracy of the model.
        """
        result = 0.
        return result

    def analyse_seasonable_variations_progress(self, aging_progress_rate):
        """
        To analyze the seasonable variations for the same person by comparing the past results on aging progress
        to define recurring pattern of changes over seasons, i.e. daily, weekly, monthly, or season.
        :param aging_progress_rate: dict, rate of aging progress over a period of time with description.
        :return: ndarray, the chart show detected recurring patterns of changes over seasons based on aging progress.
        result = Chart
        """
        result = []
        return result


class AgingEvolution:
    """
    Rate Aging Progress
    """
    at = AgingTrend()
    asv = AgingSeasonableVariator()

    def make_preprocessing(self, dataset):
        """
        To make preprocessing for initial data like cropping and aligning faces, and etc.
        :param dataset: array, with initial data.
        :return: array, with processed dataset.
        """
        result = []
        return result

    def seperate_dataset(self, processed_dataset):
        """
        To separate dataset on train_set and test_set.
        :param processed_dataset: array, with processed dataset.
        :return: train_set, test_set on which the model will be train and evaluate
        """
        result = []
        return result

    def train_model(self, train_set):
        """
        To train a model on train_set using machine learning algorithms.
        :param train_set: array, dataset on which the model will be train.
        :return: the trained model which can be used for prediction target's apparent age.
        """
        result = False
        return result

    def estimate_model(self, test_set):
        """
        To estimates the trained model on test_set to define accuracy of the model.
        :param test_set: array, dataset on which the model will be evaluate.
        :return: To estimates the trained model on test_set to define accuracy of the model.
        """
        result = 0.
        return result

    def analyse_progress_rate(self, retrieve_from, retrieve_to):
        """
        To computes the Rate of Aging Progress for the same person by comparing the past face images
        that retrieved in a database over a specified period of time.
        :param1 retrieve_from: date, the date which method have to start retrieving the past face images.
        :param2 retrieve_to: date, the date which method have to finish retrieving the past face images.
        :return: dict, rate of aging progress over a specified period of time with description.
        result = {'Became older': '1.17'}
        """
        result = {}
        return result