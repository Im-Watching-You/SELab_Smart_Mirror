"""
Date: 2019.07.10
Programmer: DH
Description: About AAA System Database and Model Monitor
"""
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
import pandas as pd


class DataBaseMonitor:
    '''
    To retrieve all kinds of table info
    '''
    def retrieve_user_table_info(self):
        '''
        To make brief information of User Table
        :return: dictionary
        '''
        um = UserManager()
        user_table = pd.DataFrame(um.retrieve_user())
        male_table = user_table.loc[(user_table['gender'] == 'male') | (user_table['gender'] == 'Male')]
        female_table = user_table.loc[(user_table['gender'] == 'female') | (user_table['gender'] == 'Female')]

        info = {'total_num_of_user':len(user_table['id']),
                'num_of_male':len(male_table['id']),
                'num_of_female':len(female_table['id']),
                'avg_age_of_user':round(user_table['age'].mean()),
                'avg_age_of_male':round(male_table['age'].mean()),
                'avg_age_of_female':round(female_table['age'].mean())}

        return info

    def retrieve_staff_table_info(self):
        '''
        To make brief information of Staff Table
        :return: dictionary
        '''
        sf = StaffManager()
        staff_table = pd.DataFrame(sf.retrieve_staff())
        male_table = staff_table.loc[(staff_table['gender'] == 'male') | (staff_table['gender'] == 'Male')]
        female_table = staff_table.loc[(staff_table['gender'] == 'female') | (staff_table['gender'] == 'Female')]

        info = {'total_num_of_staff': len(staff_table['id']),
                'num_of_male': len(male_table['id']),
                'num_of_female': len(female_table['id']),
                'avg_age_of_staff': round(staff_table['age'].mean()),
                'avg_age_of_male': round(male_table['age'].mean()),
                'avg_age_of_female': round(female_table['age'].mean())}

        return info

    def retrieve_session_table_info(self):
        '''
        To make brief information of Session Table
        :return: dictionary
        '''
        ss = Session()
        session_table = pd.DataFrame(ss.retrieve_sessions())

        value_counts = session_table['u_id'].value_counts(dropna=True)
        df_value_counts = pd.DataFrame(value_counts).reset_index()
        df_value_counts.columns = ['u_id', 'number']

        info = {'total_num_of_sessions': len(session_table['id']),
                'most_visited_user': df_value_counts.at[0, 'u_id'],
                'number_of_most_visited_count': df_value_counts.at[0, 'number'],
                'number_of_login_user': 0}

        return info

    def retrieve_photo_table_info(self):
        '''
        To make brief information of Photo Table
        :return: dictionary
        '''
        photo = Photo()
        photo_table = pd.DataFrame(photo.retrieve_photo())

        info = {'total_num_of_photos': len(photo_table['id'])}

        return info

    def retrieve_assessment_table_info(self):
        '''
        To make brief information of Assessment Table
        :return: dictionary
        '''
        ag = AgingDiagnosis()
        aging_table = pd.DataFrame(ag.retrieve_assessment_by_ids())
        ed = EmotionDiagnosis()
        emotion_table = pd.DataFrame(ed.retrieve_assessment_by_ids())
        rec = AgingRecommendation()
        recommendation_table = pd.DataFrame(rec.retrieve_assessment_by_ids())

        info = {'total_num_of_assessments': len(aging_table['id'])+
                                            len(emotion_table['id'])+
                                            len(recommendation_table['id']),
                'num_of_aging_diagnosis': len(aging_table['id']),
                'num_of_emotion_diagnosis': len(emotion_table['id']),
                'num_of_aging_recommendation': len(recommendation_table['id'])}

        return info

    def retrieve_feedback_table_info(self):
        '''
        To make brief information of Session Table
        :return: dictionary
        '''
        af = AgingFeedback()
        aging_feedback_table = pd.DataFrame(af.retrieve_feedback_by_ids())
        ef = EmotionFeedback()
        emotion_feedback_table = pd.DataFrame(ef.retrieve_feedback_by_ids())
        rf = RecommendationFeedback()
        recommend_feedback_table = pd.DataFrame(rf.retrieve_feedback_by_ids())

        info = {'total_num_of_feedbacks': len(aging_feedback_table['id'])+
                                          len(emotion_feedback_table['id'])+
                                          len(recommend_feedback_table['id']),
                'num_of_aging_feedback': len(aging_feedback_table['id']),
                'num_of_emotion_feedback': len(emotion_feedback_table['id']),
                'num_of_recommendation_feedback': len(recommend_feedback_table['id'])}

        return info

    def retrieve_training_set_table_info(self):
        '''
        To make brief information of Training Set Table
        :return:dictionary
        '''
        ts = TrainingSet()
        train_table = pd.DataFrame(ts.retrieve_training_set())

        info = {'total_num_of_training_set': len(train_table['id'])}

        return info

    def retrieve_app_table_info(self):
        '''
        To make brief information of AAAApp Table
        :return: dictionary
        '''

        ap = AAAApp()
        app_table = pd.DataFrame(ap.retrieve_app())

        info = {'total_num_of_updates_for_app': len(app_table['id']),
                'the_newest_version_of_app': app_table.sort_values(by=['version'], ascending=False)['version'].get(0)}

        return info

    def retrieve_remedy_method_table_info(self):
        '''
        To make brief information of remedy Table
        :return: dictionary
        '''

        remedy = RemedyMethod()
        remedy_table = pd.DataFrame(remedy.retrieve_all_rm())

        info = {'total_num_of_remedy_method': len(remedy_table['description'])}

        return info


class ModelMonitor:
    '''
    To retrieve aging, emotion, recommendation, face_detection model info
    '''

    def retrieve_aging_model_info(self):
        '''
        To make brief information of Aging Model Table
        :return:dictionary
        '''
        aging_model = AgingModel()
        model_table = pd.DataFrame(aging_model.retrieve_model_by_ids())
        df_id = model_table.loc[model_table['accuracy'] == model_table['accuracy'].max()].reset_index()

        info = {'total_num_of_aging_model': len(model_table['id']),
                'avg_accuracy_of_aging_model': round(model_table['accuracy'].mean()),
                'max_accuracy_of_aging_model': model_table['accuracy'].max(),
                'min_accuracy_of_aging_model': model_table['accuracy'].min(),
                'id_of_max_aging_model': df_id['id'].get(0),
                'num_of_data_of_max_aging_model': df_id['num_of_data'].get(0)}

        return info

    def retrieve_emotion_model_info(self):
        '''
        To make brief information of Emotion model Table
        :return: dictionary
        '''
        emotion_model = EmotionModel()
        model_table = pd.DataFrame(emotion_model.retrieve_model_by_ids())
        df_id = model_table.loc[model_table['accuracy'] == model_table['accuracy'].max()].reset_index()

        info = {'total_num_of_emotion_model': len(model_table['id']),
                'avg_accuracy_of_emotion_model': round(model_table['accuracy'].mean()),
                'max_accuracy_of_emotion_model': model_table['accuracy'].max(),
                'min_accuracy_of_emotion_model': model_table['accuracy'].min(),
                'id_of_max_emotion_model': df_id['id'].get(0),
                'num_of_data_of_max_emotion_model': df_id['num_of_data'].get(0)}

        return info

    def retrieve_recommendation_model_info(self):
        '''
        To make brief information of recommendation model Table
        :return: dictionary
        '''
        recommendation_model = RecommendationModel()
        model_table = pd.DataFrame(recommendation_model.retrieve_model_by_ids())
        df_id = model_table.loc[model_table['accuracy'] == model_table['accuracy'].max()].reset_index()

        info = {'total_num_of_recommendation_model': len(model_table['id']),
                'avg_accuracy_of_recommendation_model': round(model_table['accuracy'].mean()),
                'max_accuracy_of_recommendation_model': model_table['accuracy'].max(),
                'min_accuracy_of_recommendation_model': model_table['accuracy'].min(),
                'id_of_max_recommendation_model': df_id['id'].get(0),
                'num_of_data_of_max_recommendation_model': df_id['num_of_data'].get(0)}

        return info

    def retrieve_face_detection_model_info(self):
        '''
        To make brief information of face detection model Table
        :return: dictionary
        '''
        face_detection_model = FaceRecognitionModel()
        model_table = pd.DataFrame(face_detection_model.retrieve_model_by_ids())
        df_id = model_table.loc[model_table['accuracy'] == model_table['accuracy'].max()].reset_index()

        info = {'total_num_of_face_detection_model': len(model_table['id']),
                'avg_accuracy_of_face_detection_model': round(model_table['accuracy'].mean()),
                'max_accuracy_of_face_detection_model': model_table['accuracy'].max(),
                'min_accuracy_of_face_detection_model': model_table['accuracy'].min(),
                'id_of_max_face_detection_model': df_id['id'].get(0),
                'num_of_data_of_max_face_detection_model': df_id['num_of_data'].get(0)}

        return info



# Test
# db = DataBaseMonitor()
# print(db.retrieve_user_table_info())
# print(db.retrieve_staff_table_info())
# print(db.retrieve_session_table_info())
# print(db.retrieve_photo_table_info())
# print(db.retrieve_assessment_table_info())
# print(db.retrieve_feedback_table_info())
# print(db.retrieve_app_table_info())
# print(db.retrieve_training_set_table_info())
# print(db.retrieve_remedy_method_table_info())
# ml = ModelMonitor()
# print(ml.retrieve_aging_model_info())
# print(ml.retrieve_emotion_model_info())
# print(ml.retrieve_face_detection_model_info())
# print(ml.retrieve_recommendation_model_info())