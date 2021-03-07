"""
Date: 2019.07.10
Programmer: DH
Description: About AAA System Backup manager
"""
import pandas as pd
from WC.profilemanager import UserManager
from WC.profilemanager import StaffManager
from WC.session import Session
from WC.photo import Photo
from WC.assesment import AgingDiagnosis
from WC.assesment import EmotionDiagnosis
from WC.assesment import AgingRecommendation
from WC.feedback import AgingFeedback
from WC.feedback import EmotionFeedback
from WC.feedback import RecommendationFeedback
from WC.mlmodel import AgingModel
from WC.mlmodel import EmotionModel
from WC.mlmodel import FaceRecognitionModel
from WC.mlmodel import RecommendationModel
from WC.aaaapp import AAAApp
from WC.trainingset import TrainingSet
from WC.remedymethod import RemedyMethod
from distutils.dir_util import copy_tree
import os

PHOTO_PATH = r"C:\Users\SEL\PycharmProjects\SELab_Smart_Mirror\MK\CODE TEST" \
             r"\emotion_factor_analyser\5_facial_landmarks\faces"
AGE_MODEL_PATH = "."
EMOTION_MODEL_PATH = ""
RECOMMENDATION_MODEL_PATH = ""
FACE_DETECTION_PATH = ""


class BackUpManager:
    """
    To export db tables, models, photos, and load csv file to db.
    """
    def __init__(self):
        try:
            self.user = UserManager()
            self.staff = StaffManager()
            self.session = Session()
            self.photo = Photo()
            self.app = AAAApp()
            self.ag = AgingDiagnosis()
            self.ed = EmotionDiagnosis()
            self.ar = AgingRecommendation()
            self.af = AgingFeedback()
            self.ef = EmotionFeedback()
            self.rf = RecommendationFeedback()
            self.aging_model = AgingModel()
            self.emotion_model = EmotionModel()
            self.recommendation_model = RecommendationModel()
            self.face_detection_model = FaceRecognitionModel()
            self.training = TrainingSet()
            self.remedy = RemedyMethod()
        except Exception as e:
            print(e)


    def export_tables(self, save_path, *table_type):
        """
        To save model by given condition (1 ~ whole)
        :param save_path: String, directory where to save csv file
        :param table_type: String, 'user', 'staff', ‘app’, 'session', 'photo', 'assessment', 'feedback', ‘model’,
        'remedy', 'training'
        :return: Bool, True or False, shows whether operation go well or not
        """
        result = True
        if len(table_type) == 0:
            try:
                # To save all tables
                user_table = pd.DataFrame(self.user.retrieve_user())
                result = self.__save_to_csv(user_table, save_path, 'user') & result

                staff_table = pd.DataFrame(self.staff.retrieve_staff())
                result = self.__save_to_csv(staff_table, save_path, 'staff') & result

                session_table = pd.DataFrame(self.session.retrieve_sessions())
                result = self.__save_to_csv(session_table, save_path, 'session') & result

                photo_table = pd.DataFrame(self.photo.retrieve_photo())
                result = self.__save_to_csv(photo_table, save_path, 'photo') & result

                app_table = pd.DataFrame(self.app.retrieve_app())
                result = self.__save_to_csv(app_table, save_path, 'app') & result

                training_table = pd.DataFrame(self.training.retrieve_training_set())
                result = self.__save_to_csv(training_table, save_path, 'training') & result

                remedy_table = pd.DataFrame(self.remedy.retrieve_all_rm())
                result = self.__save_to_csv(remedy_table, save_path, 'remedy_method') & result

                aging_table = pd.DataFrame(self.ag.retrieve_assessment_by_ids())
                emotion_table = pd.DataFrame(self.ed.retrieve_assessment_by_ids())
                recommendation_table = pd.DataFrame(self.ar.retrieve_assessment_by_ids())
                df_assessment = aging_table
                df_assessment = df_assessment.append(emotion_table, sort=False)
                df_assessment = df_assessment.append(recommendation_table, sort=False)
                result = self.__save_to_csv(df_assessment, save_path, 'assessment') & result

                aging_feedback_table = pd.DataFrame(self.af.retrieve_feedback_by_ids())
                emotion_feedback_table = pd.DataFrame(self.ef.retrieve_feedback_by_ids())
                recommend_feedback_table = pd.DataFrame(self.rf.retrieve_feedback_by_ids())
                df_feedback = pd.DataFrame(aging_feedback_table)
                df_feedback = df_feedback.append(emotion_feedback_table, sort=False)
                df_feedback = df_feedback.append(recommend_feedback_table, sort=False)
                result = self.__save_to_csv(df_feedback, save_path, 'feedback') & result

                aging_table = pd.DataFrame(self.aging_model.retrieve_model_by_ids())
                emotion_table = pd.DataFrame(self.emotion_model.retrieve_model_by_ids())
                recommendation_table = pd.DataFrame(self.recommendation_model.retrieve_model_by_ids())
                face_table = pd.DataFrame(self.face_detection_model.retrieve_model_by_ids())
                df_model = aging_table
                df_model = df_model.append(emotion_table, sort=False)
                df_model = df_model.append(recommendation_table, sort=False)
                df_model = df_model.append(face_table, sort=False)
                result = self.__save_to_csv(df_model, save_path, 'model') & result

            except Exception as e:
                print(e)
                return False
            return result

        else:
            try:
                if 'user' in table_type:
                    user_table = pd.DataFrame(self.user.retrieve_user())
                    result = self.__save_to_csv(user_table, save_path, 'user') & result

                if 'staff' in table_type:
                    staff_table = pd.DataFrame(self.staff.retrieve_staff())
                    result = self.__save_to_csv(staff_table, save_path, 'staff') & result

                if 'session' in table_type:
                    session_table = pd.DataFrame(self.session.retrieve_sessions())
                    result = self.__save_to_csv(session_table, save_path, 'session') & result

                if 'photo' in table_type:
                    photo_table = pd.DataFrame(self.photo.retrieve_photo())
                    result = self.__save_to_csv(photo_table, save_path, 'photo') & result

                if 'assessment' in table_type:
                    aging_table = pd.DataFrame(self.ag.retrieve_assessment_by_ids())
                    emotion_table = pd.DataFrame(self.ed.retrieve_assessment_by_ids())
                    recommendation_table = pd.DataFrame(self.ar.retrieve_assessment_by_ids())
                    df_assessment = aging_table
                    df_assessment = df_assessment.append(emotion_table, sort=False)
                    df_assessment = df_assessment.append(recommendation_table, sort=False)
                    result = self.__save_to_csv(df_assessment, save_path, 'assessment') & result

                if 'feedback' in table_type:
                    aging_feedback_table = pd.DataFrame(self.af.retrieve_feedback_by_ids())
                    emotion_feedback_table = pd.DataFrame(self.ef.retrieve_feedback_by_ids())
                    recommend_feedback_table = pd.DataFrame(self.rf.retrieve_feedback_by_ids())
                    df_feedback = pd.DataFrame(aging_feedback_table)
                    df_feedback = df_feedback.append(emotion_feedback_table, sort=False)
                    df_feedback = df_feedback.append(recommend_feedback_table, sort=False)
                    result = self.__save_to_csv(df_feedback, save_path, 'feedback') & result

                if 'model' in table_type:
                    aging_table = pd.DataFrame(self.aging_model.retrieve_model_by_ids())
                    emotion_table = pd.DataFrame(self.emotion_model.retrieve_model_by_ids())
                    recommendation_table = pd.DataFrame(self.recommendation_model.retrieve_model_by_ids())
                    face_table = pd.DataFrame(self.face_detection_model.retrieve_model_by_ids())
                    df_model = aging_table
                    df_model = df_model.append(emotion_table, sort=False)
                    df_model = df_model.append(recommendation_table, sort=False)
                    df_model = df_model.append(face_table, sort=False)
                    result = self.__save_to_csv(df_model, save_path, 'model') & result

                if 'app' in table_type:
                    app_table = pd.DataFrame(self.app.retrieve_app())
                    result = self.__save_to_csv(app_table, save_path, 'app') & result

                if 'training' in table_type:
                    training_table = pd.DataFrame(self.training.retrieve_training_set())
                    result = self.__save_to_csv(training_table, save_path, 'training') & result

                if 'remedy' in table_type:
                    remedy_table = pd.DataFrame(self.remedy.retrieve_all_rm())
                    result = self.__save_to_csv(remedy_table, save_path, 'remedy_method') & result

            except Exception as e:
                print(e)
                return False
            return result

    def export_models(self, save_path, model_types):
        '''
        To save model by given condition (1 ~ whole)
        :param save_path: String, path where to save csv file
        :param model_types: String, ‘aging’, ‘emotion’, ‘recommendation’, ‘face'
        :return: boolean, True or False, shows whether operation go well or not
        '''
        model_path = {'aging': AGE_MODEL_PATH,
                      'emotion': EMOTION_MODEL_PATH,
                      'recommendation': RECOMMENDATION_MODEL_PATH,
                      'face': FACE_DETECTION_PATH}

        try:
            if os.path.isdir(save_path):
                copy_tree(model_path[model_types], save_path, update=1)
                return True
            else:
                return False
        except FileNotFoundError:
            return False
        except IOError:
            return False

    def export_photos(self, save_path, photo_path=None):
        '''
        To save photos by given condition
        :param save_path: To save photos by given condition
        :return: boolean, True or False, shows whether operation go well or not
        '''
        if photo_path is None:
            photo_path = PHOTO_PATH

        try:
            if os.path.isdir(save_path):
                copy_tree(photo_path, save_path, update=1)
                return True
            else:
                return False
        except FileNotFoundError:
            return False

    def restore_tables(self, save_path, *table_type):
        '''
        To load back up table files by given condition
        :param save_path: String, path where csv file saved
        :param table_type: String, 'user', 'staff', ‘app’, 'session', 'photo', 'assessment', 'feedback', ‘model’,
        'remedy', 'training'
        :return:True or False, shows whether operation go well or not
        '''

        result = True
        if len(table_type) == 0:
            user_table = self.__load_from_csv(save_path, 'user')

            staff_table = self.__load_from_csv(save_path, 'staff')

            session_table = self.__load_from_csv(save_path, 'session')

            photo_table = self.__load_from_csv(save_path, 'photo')

            assessment_table = self.__load_from_csv(save_path, 'assessment')

            feedback_table = self.__load_from_csv(save_path, 'feedback')

            model_table = self.__load_from_csv(save_path, 'model')

            app_table = self.__load_from_csv(save_path, 'app')

            return True

        else:

            if 'user' in table_type:
                user_table = self.__load_from_csv(save_path, 'user')

            if 'staff' in table_type:
                staff_table = self.__load_from_csv(save_path, 'staff')

            if 'session' in table_type:
                session_table = self.__load_from_csv(save_path, 'session')

            if 'photo' in table_type:
                photo_table = self.__load_from_csv(save_path, 'photo')

            if 'assessment' in table_type:
                assessment_table = self.__load_from_csv(save_path, 'assessment')

            if 'feedback' in table_type:
                feedback_table = self.__load_from_csv(save_path, 'feedback')

            if 'model' in table_type:
                model_table = self.__load_from_csv(save_path, 'model')

            if 'app' in table_type:
                app_table = self.__load_from_csv(save_path, 'app')

    def __save_to_csv(self, data_set, save_path, table_type):
        '''
        To save table to csv file
        :param data_set: dataframe, data to save
        :param save_path: String, path where to save csv file
        :param table_type: String, 'user', 'staff', ‘app’, 'session', 'photo', 'assessment', 'feedback', ‘model’,
        'remedy', 'training'
        :return: Bool, if save fails, return false
        '''
        filename = save_path + '\\' + table_type +'.csv'
        try:
            data_set.to_csv(filename, index=False)
            return True
        except FileNotFoundError:
            return False
        except IOError:
            return False

    def __load_from_csv(self, save_path, table_type):
        '''
        To load table from csv file
        :param save_path: String, path where csv file saved
        :param table_type: String, 'user', 'staff', ‘app’, 'session', 'photo', 'assessment', 'feedback', ‘model’,
        'remedy', 'training'
        :return: dataframe or False
        '''
        filename = save_path + '\\' + table_type + '.csv'
        try:
            df = pd.read_csv(filename)
            return df
        except FileNotFoundError:
            return False
        except IOError:
            return False



# Test
# bk = BackUpManager()
# print(bk.export_tables(r'C:\Users\SEL\Desktop\table', 'user', 'staff'))
# print(bk.export_tables(r'C:\Users\SEL\Desktop\table'))
# print(bk.export_photos(r'C:\Users\SEL\Desktop\photo'))
# print(bk.export_models(r'C:\Users\SEL\Desktop\model', 'aging'))

