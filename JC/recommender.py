import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer
from rake_nltk import Rake
import time
from datetime import datetime
from operator import itemgetter

from JC import key_util
from JC.database.db_manager import DataManager

from WC.feedback import RecommendationFeedback
from WC.person import User
from WC.session import Session
from WC.photo import Photo
from WC.assesment import AgingRecommendation
from WC.remedymethod import RemedyMethod
from WC.mlmodel import RecommendationModel
from WC.trainingset import *

from surprise import KNNBasic, KNNWithMeans, KNNWithZScore, KNNBaseline, SVD, SVDpp, NMF, SlopeOne, CoClustering
from surprise import Dataset
from surprise import accuracy
from surprise import dump
from surprise import Reader
from surprise.model_selection.split import train_test_split

# variables for defining training model path
PATH_DF_IMPLICIT_RATING_PREFIX = './dataset/rating/implicit/'
PATH_DF_EXPLICIT_RATING_PREFIX = './dataset/rating/explicit/'

PATH_MODEL_Cb_PREFIX = './model/cb/'
# PATH_MODEL_CF_SUFFIX = '.joblib'

# variable for indicating default dataset path
PATH_RM_DATASET = '../dataset/rm_emo_rating/wrinkle_rm.csv'
# list of recommender model types
TYPE_CF = 'CF' # collaorative filtering model type
TYPE_CB = 'CB' # content based model type
# list of similarity metrics
SIM_COSINE = 'cosine'
SIM_MSD = 'msd'
SIM_PEARSON = 'pearson'
SIM_PEARSON_BASELINE = 'pearson_baseline'

COLUMN_RATING = ['user_id', 'rm_id', 'rating']
COLUMN_REMEDY_METHOD = ['rm_id', 'rm_type', 'symptom', 'provider', 'url', 'maintain', 'improve', 'prevent', 'description', 'edit_date', 'tag']

DS_ANGRY_PATH = ''
DS_DISGUST_PATH = ''
DS_FEAR_PATH = ''
DS_HAPPY_PATH = ''
DS_SAD_PATH = '../dataset/rm_emo_rating/rm_sadness_rating, 0603a.csv'
DS_SURPRISE_PATH = ''
DS_NEUTRAL_PATH = ''

FEATURE_USER = 0

# A method for preparing(splitting) training dataset and testing dataset.
def prepare_dataset(df_ds, columns):
    # if recommeder_type == TYPE_CF:
    #     an object for parsing pre-defined dataset for constructing train/test dataset
    #     reader = Reader(sep=',', rating_scale=ratingScale)
    # elif recommeder_type == TYPE_CB:
    #     reader = Reader(sep=',')
    # import(read) data(raw data) from CSV file using Pandas
    # raw_df = pd.read_csv(df_path)
    # df = pd.read_csv(df_path, usecols=columns)
    # transform Pandas dataframe data(raw data) into readable data format for the Surprise library
    reader = Reader(sep=',')
    dataset = Dataset.load_from_df(df_ds[columns], reader=reader)
    # Train Dataset: 80%; Test Dataset: 20%
    trainset, testset = train_test_split(dataset, test_size=0.2, train_size=0.8)

    return dataset, trainset, testset

def transform_feature(feat, type_feat=FEATURE_USER):
    if type_feat == FEATURE_USER:
        current_year = datetime.now().year
        birth_year = pd.Timestamp(feat[2]).year
        feat['2'] = current_year - birth_year
        if feat[1] == 'male':
            feat[1] = 1
    return feat

def get_hist_recommendation(user, factor):
    hist_rm_list = []
    session_manager = Session()
    dict_condition_user = {'user_id': user['id']}
    db_session_list = session_manager.retrieve_sessions(dict_condition_user)
    photo_manager = Photo()
    photo_list = []
    for db_session in db_session_list:
        dict_info_session = {'session_id': db_session['id']}
        db_photo_list = photo_manager.retrieve_photo(dict_info_session)
        for db_photo in db_photo_list:
            photo_list.append(db_photo)

    recom_manager = AgingRecommendation()
    recom_list = []
    for photo in photo_list:
        dict_condition_photo = {'session_id': photo['s_id']}
        db_recom_list = recom_manager.retrieve_assessment_by_ids(dict_condition_photo)
        for db_recom in db_recom_list:
            recom_list.append(db_recom)

    # make a list of remedy methods(rm_list) for recommendation
    rm_manager = RemedyMethod()
    fb_manager = RecommendationFeedback()
    buffer_rm_id_list = []
    for recom in recom_list:
        dict_condition_recom = {'assessment_id': recom['id']}
        db_fb_list = fb_manager.retrieve_feedback_by_ids(dict_condition_recom)
        # if len(db_fb_list) > 0:
        for db_fb in db_fb_list:
            remedy_method_id = recom['result']
            if not remedy_method_id in buffer_rm_id_list:
                buffer_rm_id_list.append(remedy_method_id)
                db_rm_list = rm_manager.retrieve_remedy_method_by_id(remedy_method_id)
                # if len(db_rm_list) > 0:
                for db_rm in db_rm_list:
                    if db_rm['symptom'] == factor:
                        db_rm['rating'] = db_fb['rating']
                        db_rm['m_id'] = -1
                        hist_rm_list.append(db_rm)

    return hist_rm_list

def gen_df(ds_list, columns, path_df, df_name):
    df_ds = None
    is_created = False
    df_dict = {}
    for col in columns:
        col_data_list = []
        for ds in ds_list:
            data = ds[col]
            col_data_list.append(data)
            df_dict[col] = col_data_list

    pd_df = DataFrame(df_dict)
    result = pd_df.to_csv(path_df, sep=",", encoding='utf-8')
    # if result is None:
    #     is_created = True

    return is_created, df_ds

class BasicRecommender():

    _RATING_THRESHOLD = 3.0

    def __init__(self):
        pass

    def get_hist_rm(self, user, factor, max_num=5):
        '''
        get experienced remedy methods that the rating is larger than '_RATING_THRESHOLD'
        :param user:
        :param max_num:
        :return:
        '''
        hist_rm_list = []
        session_manager = Session()
        dict_condition_user = {'user_id': user['id']}
        db_session_list = session_manager.retrieve_sessions(dict_condition_user)
        photo_manager = Photo()
        photo_list = []
        for db_session in db_session_list:
            dict_info_session = {'session_id': db_session['id']}
            db_photo_list = photo_manager.retrieve_photo(dict_info_session)
            for db_photo in db_photo_list:
                photo_list.append(db_photo)

        recom_manager = AgingRecommendation()
        recom_list = []
        for photo in photo_list:
            dict_condition_photo = {'session_id': photo['s_id']}
            db_recom_list = recom_manager.retrieve_assessment_by_ids(dict_condition_photo)
            for db_recom in db_recom_list:
                recom_list.append(db_recom)

        # make a list of remedy methods(rm_list) for recommendation
        rm_manager = RemedyMethod()
        fb_manager = RecommendationFeedback()
        buffer_rm_id_list = []
        for recom in recom_list:
            dict_condition_recom = {'assessment_id': recom['id']}
            db_fb_list = fb_manager.retrieve_feedback_by_ids(dict_condition_recom)
            # if len(db_fb_list) > 0:
            for db_fb in db_fb_list:
                if db_fb['rating'] >= self._RATING_THRESHOLD:
                    remedy_method_id = recom['result']
                    if not remedy_method_id in buffer_rm_id_list:
                        buffer_rm_id_list.append(remedy_method_id)
                        db_rm_list = rm_manager.retrieve_remedy_method_by_id(remedy_method_id)
                        # if len(db_rm_list) > 0:
                        for db_rm in db_rm_list:
                            if db_rm['symptom'] == factor:
                                db_rm['rating'] = db_fb['rating']
                                db_rm['m_id'] = -1
                                hist_rm_list.append(db_rm)

        # sort by rating
        hist_rm_list = sorted(hist_rm_list, key=itemgetter('rating'))
        if len(hist_rm_list) > max_num:
            hist_rm_list = hist_rm_list[0:5]

        return hist_rm_list

    def get_pref_rm(self, user, factor, max_num=5):
        '''
        recommend top n remedy methods based on user defined preference
        :param factor: aging factor or emotion factor
        :param max_num:
        :return:
        '''
        pref_rm_list = []

        i = 5

        while len(pref_rm_list) == 0:
            pref_age_from = user['age'] - i
            pref_age_to = user['age'] + i
            pref_gender = user['gender']

            # user_manager = UserManager()
            user_manager = User()
            dict_user_group_pref = {"from:": pref_age_from, "to": pref_age_to, "gender": pref_gender}
            db_user_list = user_manager.retrieve_user_by_age_gender(dict_user_group_pref)
            # print("db_user_list: ", len(db_user_list))

            # remove the current user who is using the AAA system
            for db_user in db_user_list:
                if db_user['id'] == user['id']:
                    db_user_list.remove(db_user)
                    break

            # retrieve all session using users(db_user_list)
            session_manager = Session()
            session_list = []
            for db_user in db_user_list:
                dict_user = {"user_id": db_user['id']}
                db_session = session_manager.retrieve_sessions(dict_user)
                for sessssion in db_session:
                    session_list.append(sessssion)
            # print("session_list: ", len(session_list))

            # retrieve all photos using sessions(sess_list)
            photo_manager = Photo()
            photo_list = []
            for session in session_list:
                dict_session = {"session_id": session['id']}
                db_photo_list = photo_manager.retrieve_photo(dict_session)
                for db_photo in db_photo_list:
                    photo_list.append(db_photo)
            # print("photo_list: ", len(photo_list))

            # retrieve all recommendations using photos(photo_list)
            recom_manager = AgingRecommendation()
            recom_list = []
            for photo in photo_list:
                dict_condition_photo = {'session_id': photo['s_id']}
                db_recom_list = recom_manager.retrieve_assessment_by_ids(dict_condition_photo)
                # print("db_recom_list: ", len(db_recom_list))
                for db_recom in db_recom_list:
                    recom_list.append(db_recom)
            # print("recom_list: ", len(recom_list))

            # make a preference recommendations(pref_recom_list) that each feedback rating is larger than 'pref_rating'
            fb_manager = RecommendationFeedback()
            pref_recom_list = []
            for recom in recom_list:
                dict_condition_recom = {'assessment_id': recom['id']}
                db_fb_list = fb_manager.retrieve_feedback_by_ids(dict_condition_recom)
                for db_fb in db_fb_list:
                    if db_fb['rating'] >= 4.0:
                        recom['rating'] = db_fb['rating']
                        pref_recom_list.append(recom)
            # print("pref_recom_list: ", len(pref_recom_list))

            # retrieve remedy methods based on user preferences('pref_tag', 'pref_provider', 'symptom')
            remedy_method = RemedyMethod()
            buffer_pref_rm_id_list = []
            for pref_recom in pref_recom_list:
                db_rm_list = remedy_method.retrieve_remedy_method_by_id(pref_recom['result'])
                for db_rm in db_rm_list:
                    if db_rm['symptom'] == factor:
                        if not db_rm['rm_id'] in buffer_pref_rm_id_list:
                            buffer_pref_rm_id_list.append(db_rm['rm_id'])
                            db_rm['rating'] = pref_recom['rating']
                            db_rm['m_id'] = pref_recom['m_id']
                            pref_rm_list.append(db_rm)
            # print("pref_rm_list: ", pref_rm_list)

            i = i + 1
            # print("i: ", i)

        # print("End While loop !")
        pref_rm_list = sorted(pref_rm_list, key=itemgetter('rating'))
        if len(pref_rm_list) > max_num:
            pref_rm_list = pref_rm_list[0:5]

        return pref_rm_list

class RatingPredictor():
    # variables for indicating the collaborative filtering algorithm names
    CF_KNN_KB = 'cf_knnbasic'
    CF_KNN_KWM = 'cf_knnwithmeans'
    CF_KNN_KWZ = 'cf_knnwithZScore'
    CF_KNN_KBL = 'cf_knnbaseline'
    CF_MF_SVD = 'cf_svd'
    CF_MF_SVDPP = 'cf_svdpp'  # SVD++
    CF_MF_NMF = 'cf_nmf'
    CF_SO = 'cf_slopeone'
    CF_CC = 'cf_coclustering'
    ENS_PREDICTOR = 'ens_predictor'
    ALGO_NAME_LIST = [CF_KNN_KB, CF_KNN_KWM, CF_KNN_KWZ, CF_KNN_KBL, CF_MF_SVD, CF_MF_SVDPP, CF_MF_NMF, CF_SO, CF_CC]
    # variables for indicating rating prediction model's 'model_type'(DB column)
    MODEL_AGING_WRINKLE = 'model_rating_wrinkle'
    MODEL_AGING_HARILOSS = 'model_rating_hair_loss'
    MODEL_AGING_AGESPOT = 'model_rating_age_spot'
    MODEL_AGING_EYEBAG = 'model_rating_eye_bag'
    MODEL_AGING_SKINTONE = 'model_rating_skin_tone'
    MODEL_AGING_UNEVENSKINTONE = 'model_rating_uneven_skin_tone'
    MODEL_AGING_BALDNESS = 'model_rating_baldness'
    MODEL_AGING_SAGGING = 'model_rating_sagging'
    # variables for specifying path of training model(recommender)
    _PATH_MODEL_CF_PREFIX = './model/cf/'
    _PATH_MODEL_CF_SUFFIX = '.joblib'
    # variables on threshold value
    RATING_THRESHOLD = 3.0
    RMSE_THRESHOLD = 1
    #variables for indicating '.csv' file directory
    PATH_PREFIX_RATING_DF = '../dataset/rating/'
    COLUMN_RATING = ['user_id', 'rm_id', 'rating']
    # variables for indicating 'key' of dictionary
    PERFORMANCE_RMSE = 'rmse'
    PERFORMANCE_MAE = 'mae'
    # global variable for reserving all prediction models using different CF algorithms
    PREDICTOR_LIST = []

    def __init__(self, algo_name=None, sim_metric=SIM_MSD, user_based=False):
        if algo_name is not None:
            self.algo_name = algo_name # one of algorithms from SurPRIZE library
        else:
            self.algo_name = self.CF_KNN_KB

        self.algo = None
        self.model_name = None
        self.train_set = None
        self.test_set = None
        self.rmse = None
        self.accuracy = None
        self.saved_path = None
        self.training_set = None

        if user_based is True:
            sim = {'name': sim_metric, 'user_based': True}
        else:
            sim = {'name': sim_metric, 'user_based': False}

        if self.algo_name == self.CF_KNN_KB:
            self.algo = KNNBasic(sim_options=sim, verbose=False)
            self.model_name = self.CF_KNN_KB + '_' + str(round(time.time()))
        elif self.algo_name == self.CF_KNN_KWM:
            self.algo = KNNWithMeans(sim_options=sim, verbose=False)
            self.model_name = self.CF_KNN_KWM + '_' + str(round(time.time()))
        elif self.algo_name == self.CF_KNN_KWZ:
            self.algo = KNNWithZScore(sim_options=sim, verbose=False)
            self.model_name = self.CF_KNN_KWZ + '_' + str(round(time.time()))
        elif self.algo_name == self.CF_KNN_KBL:
            self.algo = KNNBaseline(sim_options=sim, verbose=False)
            self.model_name = self.CF_KNN_KBL + '_' + str(round(time.time()))
        elif self.algo_name == self.CF_MF_SVD:
            self.algo = SVD()
            self.model_name = self.CF_MF_SVD + '_' + str(round(time.time()))
        elif self.algo_name == self.CF_MF_SVDPP:
            self.algo = SVDpp()
            self.model_name = self.CF_MF_SVDPP + '_' + str(round(time.time()))
        elif self.algo_name == self.CF_MF_NMF:
            self.algo = NMF()
            self.model_name = self.CF_MF_NMF + '_' + str(round(time.time()))
        elif self.algo_name == self.CF_SO:
            self.algo = SlopeOne()
            self.model_name = self.CF_SO + '_' + str(round(time.time()))
        elif self.algo_name == self.CF_CC:
            self.algo = CoClustering(verbose=False)
            self.model_name = self.CF_CC + '_' + str(round(time.time()))

    def _meas_corrcoef(self, pred_label, real_label):
        list_df = []
        for i in range(0, len(real_label)):
            list_df.append(pred_label[i].copy())
            list_df[i].append(real_label[i])

        # measure correlation-coefficient to select relevant 'recommender models'(algorithms)
        list_corrcoef = np.array(list_df)
        dict_ds = {index: list_corrcoef[:, index] for index in range(0, len(list_corrcoef[1]))}
        df_ds = DataFrame(dict_ds)

        return df_ds

    def get_rating_df(self, factor):
        path_df = None
        if factor == key_util.SYMPTOM_AGING_WRINKLE:
            path_df = './dataset/rating_wrinkle.csv'
        elif factor == key_util.SYMPTOM_AGING_BALDNESS:
            path_df = './dataset/rating_baldness.csv'
        elif factor == key_util.SYMPTOM_AGING_AGESPOT:
            path_df = './dataset/rating_age_spot.csv'

        df_rating = pd.read_csv(path_df, usecols=COLUMN_RATING)

        return df_rating

    # common interface of the MLModel class
    def train(self, train_set, sim_metric=SIM_MSD, user_based=False):
        self.train_set = train_set
        self.algo.fit(train_set)

        return self

    # common interface of the MLModel class
    def estimate_model(self, test_set, model=None):
        self.test_set = test_set
        if model is not None:
            self = model

        self.predictions = self.algo.test(test_set)
        self.rmse = accuracy.rmse(self.predictions, verbose=False)
        self.mae = accuracy.mae(self.predictions, verbose=False)

        label_real = []
        label_pred = []
        for ts in test_set:
            label_real.append(ts[2])
            pred_rating = round(self.pred_rating(user_id=ts[0], rm_id=ts[1]))
            label_pred.append(pred_rating)

        self.accuracy = round(accuracy_score(label_real, label_pred) * 100, 10)

        return self.accuracy

    def train_est_model(self, train_set, test_set):
        '''
        generate several rating prediction models and return the highest accuracy model which the RMSE value is lowest one
        :param train_set:
        :param sim_metric:
        :param user_based:
        :return:
        '''
        predictor_list = []
        for algo_name in self.ALGO_NAME_LIST:
            buffer_predictor = self.__init__(algo_name=algo_name)
            buffer_predictor.algo.fit(train_set)
            buffer_predictor.estimate_model(test_set)
            dict_buffer_predictor = {}
            dict_buffer_predictor['model_name'] = buffer_predictor.model_name
            dict_buffer_predictor['predictor'] = buffer_predictor
            buffer_predictor.saved_path = buffer_predictor._PATH_MODEL_CF_PREFIX + buffer_predictor.model_name + buffer_predictor._PATH_MODEL_CF_SUFFIX
            dict_buffer_predictor['saved_path'] = buffer_predictor.saved_path
            dict_buffer_predictor['rmse'] = buffer_predictor.rmse
            dict_buffer_predictor['mae'] = buffer_predictor.mae
            dict_buffer_predictor['accuracy'] = buffer_predictor.accuracy
            predictor_list.append(dict_buffer_predictor)

        self.PREDICTOR_LIST = sorted(predictor_list, key=itemgetter('rmse'))
        self = self.PREDICTOR_LIST[0]['predictor']
        self.save_model()

        return self

    # common interface of the MLModel class
    def divide_dataset(self, df_ds, columns):
        reader = Reader(sep=',')
        data_set = Dataset.load_from_df(df_ds[columns], reader=reader)
        # Train Dataset: 80%; Test Dataset: 20%
        train_set, test_set = train_test_split(data_set, test_size=0.2)

        return train_set, test_set

    # common interface of the MLModel class
    def purify_data(self, data_set):

        return False

    # save trained rating prediction model
    def save_model(self):
        self.saved_path = self._PATH_MODEL_CF_PREFIX + self.model_name + self._PATH_MODEL_CF_SUFFIX
        if self.rmse is not None and self.mae is not None:
            dump.dump(self.saved_path, predictions=self.predictions, algo=self.algo, verbose=1)
        else:
            dump.dump(self.saved_path, algo=self.algo, verbose=1)

    # load existing rating prediction model
    def load_model(self, saved_path):
        '''
        To load a pre-trained model
        (exception handling required for failing to load exiting algorithm)
        '''
        train_model = dump.load(saved_path)
        self.predictions = train_model[0]
        self.algo = train_model[1]

        return self.algo

    def pred_rating(self, user_id, rm_id, real_rating=None):
        '''
        To predict rating using the current model
        pred_list: [User ID,
                    Item ID(Remedy Method ID),
                    Input Rating,
                    Predcited Rating,
                    Detail Information for Learning]
        '''
        prediction = self.algo.predict(user_id, rm_id, r_ui=real_rating)
        rating = round(prediction[3])

        return rating

    # def train_ens_model(self, train_set):
    #     '''
    #     train a linear regressor [[pred_rate(knn_basic), pred_rate(knn_kwz), ...], [real_rate, ...]]
    #     :param train_set:
    #     :param test_set:
    #     :return:
    #     '''
    #     self.ens_model_name = self.ENS_PREDICTOR + '_' + str(round(time.time()))
    #     self.ens_saved_path = self._PATH_MODEL_CF_PREFIX + self.ens_model_name + self._PATH_MODEL_CF_SUFFIX
    #     self.ens_recommender = []
    #     for model_name in self.MODEL_NAME_LIST:
    #         buffer_recommender = CfRecommender(model_name)
    #         buffer_recommender.train(train_set)
    #         self.ens_recommender.append(buffer_recommender)
    #
    #     # features are predicted ratings and labels are the real ratings of the 'train_set'
    #     reg_train_feat = []
    #     reg_train_label = []
    #     for ts in train_set.all_ratings():
    #         reg_train_label.append(ts[2])
    #         feat_value = []
    #         for recommender in self.ens_recommender:
    #             pred_value = recommender.model.predict(ts[0], ts[1])
    #             feat_value.append(pred_value[3])
    #         reg_train_feat.append(feat_value)
    #
    #     # transform python list to numpy array for training linear regression
    #     reg_train_feat = np.array(reg_train_feat)
    #     reg_train_label = np.array(reg_train_label)
    #     self.ens_model = LinearRegression().fit(reg_train_feat, reg_train_label)
    #
    # def estimate_ens_model(self, test_set):
    #     '''
    #     estimate(evaluate) ensemble model(linear regressor)
    #     :param test_set:
    #     :return:
    #     '''
    #     reg_test_feat = []
    #     reg_test_label = []
    #     for ts in test_set:
    #         reg_test_label.append(ts[2])
    #         feat_value = []
    #         for recommender in self.ens_recommender:
    #             pred_value = recommender.model.predict(ts[0], ts[1])
    #             feat_value.append(pred_value[3])
    #         reg_test_feat.append(feat_value)
    #
    #     # print("reg_test_feat: ", reg_test_feat)
    #     # print("reg_test_label: ", reg_test_label)
    #     # transform python list to numpy array for testing linear regression
    #     reg_test_feat = np.array(reg_test_feat)
    #     reg_test_label = np.array(reg_test_label)
    #     pred_label = self.ens_model.predict(reg_test_feat)
    #     # print("reg_test_label: ", reg_test_label)
    #     self.ens_rmse = mean_squared_error(reg_test_label, pred_label)
    #     print("Ensemble RMSE: ", self.ens_rmse)
    #
    # def pred_rating_ens(self, user_id, rm_id):
    #     pred_feat = []
    #     feat_value = []
    #     for recommender in self.ens_recommender:
    #         feat_pred_value = recommender.model.predict(user_id, rm_id)
    #         feat_value.append(feat_pred_value[3])
    #
    #     pred_feat.append(feat_value)
    #     pred_feat = np.array(pred_feat)
    #     rating = self.ens_model.predict(pred_feat)
    #
    #     return rating

class RecommendationController():

    def __init__(self, user=None, assessment=None, photo=None, user_pref=None):
        self._buffer_rm_list = []
        self._hist_rm_list = []
        if user is not None:
            self.user = user
        else:
            self.user = None

        if assessment is not None:
            self.assessment = assessment
        else:
            self.assessment = None

        if user_pref is not None:
            self.user_pref = user_pref
        else:
            self.user_pref = None

        if photo is not None:
            self.photo = photo
        else:
            self.photo = photo

    def check_history(self, user, factor):
        hist_rm_list = get_hist_recommendation(user, factor)
        if len(hist_rm_list) > 0:
            return True

        return False

    def _get_common_rm(self, recom_list):
        rm_list = []
        for i in range(0, len(recom_list)):
            k = i + 1
            rm_id = recom_list[i]['rm_id']
            log = -1
            for j in range(k, len(recom_list)):
                if rm_id == recom_list[j]['rm_id']:
                    recom_list[j]['rm_id'] = -1
                    if log == -1 and rm_id > -1:
                        rm_list.append(recom_list[i].copy())
                        log = 1
            recom_list[i]['rm_id'] = -1

        return rm_list

    def _get_basic_rm(self, user, factor):
        br = BasicRecommender()
        pref_rm_list = br.get_pref_rm(user, factor)
        for pref_rm in pref_rm_list:
            self._buffer_rm_list.append(pref_rm)

    def _get_cf_rm(self, user, factor):
        '''
        generate a number of remedy methods using 'Collaborative Filtering approach using the
        RatingPredictor and add to the candidate 'buffer_rm_list' for further hybrid recommendation
        :return: no return value
        '''
        rating_predictor = RatingPredictor()
        cf_rm_list = []
        # recommend remedy methods to the 'factor'
        df_rating = rating_predictor.get_rating_df(factor)
        df_list = df_rating.values.tolist()
        hist_rm_list = []
        user_id = user['id']
        for df in df_list:
            if user_id == df[0]:
                hist_rm_list.append(df)
        # filtering history(experienced) remedy method list
        rm_manager = RemedyMethod()
        rms = rm_manager.retrieve_all_rm(factor)
        for hist_rm in hist_rm_list:
            for rm in rms:
                if rm != -1:
                    if rm['rm_id'] == hist_rm[1]:
                        index = rms.index(rm)
                        rms[index] = -1
        # retrieve rating prediction model on remedy methods for the 'factor'
        recom_model = RecommendationModel()
        model_latest_id = recom_model.retrieve_latest_model_id(key_util.MODEL_AGING_BALDNESS)
        dict_model_ids = {'id': model_latest_id}
        db_pred_rating_model = recom_model.retrieve_model_by_ids(dict_model_ids)
        saved_path = db_pred_rating_model[0]['saved_path']
        rating_predictor.load_model(saved_path)
        # predict ratings of not yet experienced remedy methods
        for rm in rms:
            if rm != -1:
                rm_id = rm['rm_id']
                rating = rating_predictor.pred_rating(user_id, rm_id)
                if rating >= rating_predictor.RATING_THRESHOLD:
                    rm['pred_rating'] = rating
                    rm['m_id'] = model_latest_id
                    cf_rm_list.append(rm)
        # add cf-based retrieved rememdy methods to candidate 'buffer_rm_list'
        if len(cf_rm_list) > 0:
            cf_rm_list = sorted(cf_rm_list, key=itemgetter('pred_rating'))
            for cf_rm in cf_rm_list:
                self._buffer_rm_list.append(cf_rm)

        # # testing code to print CF algorithm based recommendation
        # for cf_rm in cf_rm_list:
        #     print("CF_RM: ", cf_rm)

    def recom_rm(self, user, assessment, photo):
        recom_dict = {}
        # recommend remedy methods to all of the factors
        for key in assessment:
            factors = None
            if key == 'age_ft':
                factors = assessment[key]
            elif key == 'emotion_ft':
                factors = assessment[key]
            # recommend remedy methods for each factor
            if factors is not None:
                for f in factors:
                    factor = list(f.keys())[0]
                    if self.check_history(user, factor):
                        self._get_cf_rm(user, factor)
                    else:
                        # print("Welcome {} ! It's your first time to use the AAA system !".format(user['first_name']))
                        self._get_basic_rm(user, factor)
                    recom_rm_list = self._buffer_rm_list
                    # after finishing the hybrid recommendation, remove duplicated remedy methods, and generate a common list(rm_list)
                    # recom_rm_list = self._get_common_rm(self._buffer_rm_list)
                    # for hist_rm in self._hist_rm_list:
                    #     recom_rm_list.append(hist_rm)
                    recom_manager = AgingRecommendation()
                    # record recommendations to DB
                    for recom_rm in recom_rm_list:
                        if key == 'age_ft':
                            ass = assessment['age']
                        elif key == 'emotion_ft':
                            ass = assessment['emotion']
                        dict_info_recom = {'photo_id': ass['p_id'], 'result': recom_rm['rm_id'], 'session_id': ass['s_id'], 'model_id': recom_rm['m_id']} # photo id is dummy data
                        recom_manager.register_assessment(dict_info_recom)

                    recom_dict[factor] = recom_rm_list

        return recom_dict

# # test code for predicted rating based recommendation
# if __name__ == '__main__':
#     user = {'id': 294, 'gender': 'Male', 'first_name': 'Zhenzhe', 'last_name': 'Piao', 'phone_number': '01094068826', 'email': 'jincheul826@gmail.com', 'age': 32}
#     photo = {'id': 1, 's_id': 619}
#     assessment = {'age': {'id': 514, 'assess_type': 'aging_diagnosis', 'm_id': 148, 'p_id': 1, 's_id': 619, 'result': 32, 'recorded_date': '2019-07-09 15:11:38'},
#                   'age_ft': [{'wrinkle': 32}]
#                   }
#     recom = RecommendationController()
#     recom_dict = recom.recom_rm(user, assessment, photo)
#     print("{}, we recommend some remedy methods for your {} aging factor.".format(user['first_name'], "'wrinkle'"))
#     print("Please try the following guides !")
#     i = 0
#     for rm in recom_dict['wrinkle']:
#         i = i +1
#         print("{}. {}".format(i, rm['description']))

# # test code on recommendations for a new user
# if __name__ == '__main__':
#     user = {'id': 338, 'gender': 'Male', 'first_name': 'Lora', 'last_name': 'Kim', 'phone_number': '01013548794', 'email': 'Lora@gmail.com', 'age': 32}
#     photo = {'id': 1, 's_id': 3625}
#     assessment = {'age': {'id': 1057, 'assess_type': 'aging_diagnosis', 'm_id': 1, 'p_id': 1, 's_id': 3625, 'result': 34, 'recorded_date': '2019-07-18 20:04:40'},
#                   'age_ft': [{'wrinkle': 32}]
#                   }
#     recom = RecommendationController()
#     recom_dict = recom.recom_rm(user, assessment, photo)
#     print("{}, we recommend some remedy methods for your {} aging factor.".format(user['first_name'], "'wrinkle'"))
#     print("Please try the following guides !")
#     i = 0
#     for rm in recom_dict['wrinkle']:
#         i = i + 1
#         print("{}. {}".format(i, rm['description']))

# # test code for training, estimating, and saving a model
# if __name__ == '__main__':
#     rate_predictor = RatingPredictor()
#     df_rating = rate_predictor.get_rating_df('baldness')
#     train_set, test_set = rate_predictor.divide_dataset(df_rating, columns=rate_predictor.COLUMN_RATING)
#     # df_list = df_rating.values.tolist()
#     rate_predictor.train(train_set)
#     rate_predictor.estimate_model(test_set)
#     rate_predictor.save_model()
#     print("Model RMSE: ", rate_predictor.rmse)

# # test code for history recommendation retrieve
# if __name__ == '__main__':
#     print(get_hist_recommendation({'id': 294}))

# # test code for BR recommendation
# if __name__ == '__main__':
#     user = {'id': 294, 'age': 32, 'gender': 'male'}
#     age = 32
#     factor = 'wrinkle'
#     user_pref = {key_util.KEY_PREF_AGE_FROM: age-2,
#                  key_util.KEY_PREF_AGE_TO: age+2,
#                  key_util.KEY_PREF_GENDER: 'male'}
#     br = BasicRecommender()
#     pref_rm_list = br.get_pref_rm(user, factor)
#
#     for pref_rm in pref_rm_list:
#         print("pref_rm: ", pref_rm)