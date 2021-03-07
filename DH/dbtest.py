from WC.person import User
from WC.person import Staff
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
import datetime
import pandas as pd

# user = User()
# inputlist = []
# print(type(inputlist))
# u_input = {'first_name' : 'Joe', 'last_name':'Evans'}
# inputlist.append(u_input)
# u_input = {'last_name' : 'Li'}
# inputlist.append(u_input)
# u_input = {'user_id' : '2', 'first_name':'Joe', 'last_name':'Evan'}
# inputlist.append(u_input)
# u_input = {'phone_number' : '01090193991'}
# inputlist.append(u_input)
# u_input = {'joined_date' : '2019-06-30 16:17'}
# inputlist.append(u_input)
# # u_input = {'end_joined' : '20190630'}
# # inputlist.append(u_input15)
# u_input= {'order_by':'rand'}
# inputlist.append(u_input)
# u_input = {'order_by':'male_first'}
# inputlist.append(u_input)
# u_input = {'order_by':'female_first'}
# inputlist.append(u_input)
# u_input= {'order_by':'birth_asc'}
# inputlist.append(u_input)
# u_input = {'order_by':'birth_desc'}
# inputlist.append(u_input)
# u_input= {'order_by':'joined_date_asc'}
# inputlist.append(u_input)
# u_input={'order_by':'joined_date_desc'}
# inputlist.append(u_input)
# u_jdate = datetime(2019, 6, 24, 16, 17)

# u_input = {'password':'123435', 'gender':'female', 'birth_date':'19510101','first_name':'Harry', 'last_name':'John',
#            'phone_number':'01012341234','email':'test3@naver.com','joined_date':'20190701', 'age': '70'}
# print(user.retrieve_user({'user_id':1}))
# print(user.update_user_profile(1, u_input))
# print(user.retrieve_user({'user_id':1}))
#
# staff = Staff()
# print(staff.retrieve_staff({'user_id':2}))
# print(staff.update_staff_profile(2, u_input))
# print(staff.retrieve_staff({'user_id':2}))
# userm = UserManager()
# staffm = StaffManager()

# user. register_user ({"password": 'test1234', "gender": 'male', "birth_date": '19940915',
# "first_name": 'Woochan', 'last_name': 'Park', 'phone_number': '010000000000', 'email': 'tes@test.com'
#                       ,'ap_id':'1'})
# print(user.retrieve_user({"password": 'test1234', "gender": 'male', "birth_date": '19940915',
# "first_name": 'Woochan', 'last_name': 'Park', 'phone_number': '010000000000', 'email': 'tes@test.com'
#                       ,'ap_id':'1'}))

# staff = Staff ()
# staff. register_staff ({"password":'test123', "gender": 'male', "birth_date": '19940915',
# "first_name": 'Woochan', 'last_name': 'Park', 'phone_number': '01050005000',
# 'email': 'test1299875@test.co'})
# print(pd.DataFrame(staff.retrieve_staff({"password":'test123', "gender": 'male', "birth_date": '19940915',
# "first_name": 'Woochan', 'last_name': 'Park', 'phone_number': '01050005000',
# 'email': 'test1299875@test.co'})))
# print(pd.DataFrame(staff.retrieve_staff({'gender':'male'})))



# ad = Diagnosis()
# ad.register_assessment({"model_id": 1, "photo_id": '100', "session_id": 30, "result": 30})
# print(ad.retrieve_assessment_by_ids({'model_id':1, 'session_id':30}))

# print(ad.retrieve_assessment_by_ids({'id':37}))
# update_input = { 'm_id':'142', 'p_id':'30',
#                 'recorded_date':'20190710', 'result':'40', 's_id':'100'}
# print(ad.update_assessment(19, update_input))
# print(ad.retrieve_assessment_by_ids({'id':37}))
# ed = EmotionDiagnosis()
# print(ed.retrieve_assessment_by_ids({'id':37}))
# update_input = {'m_id':'139', 'p_id':'29',
#                 'recorded_date':'20190710', 'result':'41', 's_id':'101'}
# print(ed.update_assessment(37, update_input))
# print(ed.retrieve_assessment_by_ids({'id':37}))
#
# ar = AgingRecommendation()
# print(ar.retrieve_assessment_by_ids({'id':100}))
# update_input = {'m_id':'140', 'p_id':'30',
#                 'recorded_date':'20190710', 'result':'40', 's_id':'100'}
# print(ar.update_assessment(100, update_input))
# print(ar.retrieve_assessment_by_ids({'id':100}))

# age_input = {'from':10,'to':20,'gender':'male'}
# print(pd.DataFrame(user.retrieve_user_by_age_gender(age_input)))
# print(type(user.retrieve_user_by_age_gender(age_input)))
# age_input = {'to':20}
# print(pd.DataFrame(user.retrieve_user_by_age_gender(age_input)))
# print(type(user.retrieve_user_by_age_gender(age_input)))
# age_input = {'from':50}
# print(pd.DataFrame(user.retrieve_user_by_age_gender(age_input)))
# print(type(user.retrieve_user_by_age_gender(age_input)))
# age_input = {'gender':'female'}
# print(pd.DataFrame(user.retrieve_user_by_age_gender(age_input)))
# print(type(user.retrieve_user_by_age_gender(age_input)))
# age_input = {'from':100}
# print(pd.DataFrame(user.retrieve_user_by_age_gender(age_input)))
# print(type(user.retrieve_user_by_age_gender(age_input)))
# age_input = {'from':'10'}
# print(pd.DataFrame(user.retrieve_user_by_age_gender(age_input)))
# print(type(user.retrieve_user_by_age_gender(age_input)))
#
# aging_assessment = AgingDiagnosis()
# print(pd.DataFrame(aging_assessment.retrieve_assessments({'end_date':'20190709'})))
#
# age = AgingFeedback()
# print(age.register_feedback({"assessment_id": 30, "model_id": 3, "rating": 5}))
# print(age.retrieve_feedback_by_ids({'model_id':3, 'id':30}))
# f_input = {'as_id':'10', 'm_id':'19', 'rated_date':'20190404','rating':'1'}
# print(age.retrieve_feedback_by_ids({'id':70}))
# # print(age.update_feedback(70, f_input))
# print(age.retrieve_feedback_by_ids({'id':70}))
#
# emotion = EmotionFeedback()
# print(emotion.retrieve_feedback_by_ids({'id':3}))
# # print(emotion.update_feedback(3, f_input))
# print(emotion.retrieve_feedback_by_ids({'id':3}))
#
# recommend = RecommendationFeedback()
# print(recommend.retrieve_feedback_by_ids({'id':90}))
# # print(recommend.update_feedback(90, f_input))
# print(recommend.retrieve_feedback_by_ids({'id':90}))
#




# recommendation_assessment = EmotionDiagnosis()
# as_list = []
# as_input = {'id':40}
# print(pd.DataFrame(recommendation_assessment.retrieve_assessment_by_ids(as_input)))
# print(type(recommendation_assessment.retrieve_assessment_by_ids(as_input)))
# as_input = {'id':40, 'model_id':100}
# print(pd.DataFrame(recommendation_assessment.retrieve_assessment_by_ids(as_input)))
# print(type(recommendation_assessment.retrieve_assessment_by_ids(as_input)))
# as_input = {'id':40,'assessment_id':114}
# print(pd.DataFrame(recommendation_assessment.retrieve_assessment_by_ids(as_input)))
# print(type(recommendation_assessment.retrieve_assessment_by_ids(as_input)))
#
# print(pd.DataFrame(recommendation_assessment.retrieve_assessment_by_rating({'min_rating':4, 'sorting_order':'DESC'})))
# print(recommendation_assessment.retrieve_assessment_by_rating({'min_rating':4, 'soring_order':'ASC'}))
# print(pd.DataFrame(recommendation_assessment.retrieve_assessment_by_rating({'max_rating':3, 'min_rating':2})))
#
# session = Session()
# print((session.retrieve_sessions({'user_id':294})))
# print(session.retrieve_active_session_id(294))
# print(session.update_action_log(294, 'sss'))
# print(session.retrieve_sessions({'user_id':294}))
# print(pd.DataFrame(session.retrieve_sessions({'start_date':'20190101'})))
# print(type(session.retrieve_sessions({'start_date':'20190101'})))
# print(pd.DataFrame(session.retrieve_sessions({'end_date':'20180103'})))
# print(type(session.retrieve_sessions({'end_date':'20180104'})))
# print(pd.DataFrame(session.retrieve_sessions({'start_date':'20190705','duration':'a week'})))
# print(type(session.retrieve_sessions({'start_date':'20190705','duration':'a week'})))
#
# print(session.retrieve_active_session_id(1))
# print(session.retrieve_active_session_id(('1')))
# print(session.retrieve_active_session_id(('a')))
#
# photo = Photo()
# p_in = {'photo_id':'10', 'session_id':'68'}
# print(pd.DataFrame(photo.retrieve_photo(p_in)))
# u_input = {'photo_id':10, 'session_id':68, 'saved_path':'/root/22'}
# print(photo.update_photo(u_input))
# print(photo.retrieve_photo({'photo_id':'10', 'session_id':'68'}))
# print(type(pd.DataFrame(photo.retrieve_photo(p_in))))
# p_in = {'photo_id' : 10, 'end_date':'20190715'}
# print(pd.DataFrame(photo.retrieve_photo(p_in)))
# print(type(photo.retrieve_photo(p_in)))
# p_in = {'start_date':'20180701'}
# print(pd.DataFrame(photo.retrieve_photo(p_in)))
# print(type(pd.DataFrame(photo.retrieve_photo(p_in))))
# p_in = {'end_date':'20180701'}
# print(pd.DataFrame(photo.retrieve_photo(p_in)))
# print(type(pd.DataFrame(photo.retrieve_photo(p_in))))
# print(photo.retrieve_latest_photo_id(530))
# print(type(photo.retrieve_latest_photo_id(10)))
#
# age = AgingModel()
# m_in = {'id':'101'}
# print((age.retrieve_model_by_ids(m_in)))
# u_input = {'accuracy':'50', 'last_updated_date':'20190710', 'num_of_data':'1000',
#            'released_date':'20190705', 's_id':'1', 'saved_path':'/root9999', 't_id':'50'}
# print(age.update_model(101, u_input))
# print(age.retrieve_model_by_ids(m_in))
#
# emotion = EmotionModel()
# m_in = {'id':'101'}
# print((emotion.retrieve_model_by_ids(m_in)))
# u_input = {'accuracy':'50', 'last_updated_date':'20190710', 'num_of_data':'1000',
#            'released_date':'20190705', 's_id':'1', 'saved_path':'/root9999', 't_id':'50'}
# print(emotion.update_model(101, u_input))
# print(emotion.retrieve_model_by_ids(m_in))
#
# rec= RecommendationModel()
# m_in = {'id':'101'}
# print((rec.retrieve_model_by_ids(m_in)))
# u_input = {'accuracy':'50', 'last_updated_date':'20190710', 'num_of_data':'1000',
#            'released_date':'20190705', 's_id':'1', 'saved_path':'/root9999', 't_id':'50'}
# print(rec.update_model('aging_model',101, u_input))
# print(rec.retrieve_model_by_ids(m_in))
#
# face = FaceRecognitionModel()
# m_in = {'id':'101'}
# print((face.retrieve_model_by_ids(m_in)))
# u_input = {'accuracy':'50', 'last_updated_date':'20190710', 'num_of_data':'1000',
#            'released_date':'20190705', 's_id':'1', 'saved_path':'/root9999', 't_id':'50'}
# print(face.update_model(101, u_input))
# print(face.retrieve_model_by_ids(m_in))
#
# app = AAAApp()
# print(app.retrieve_app({'id':1}))
# print(app.update_app({'id':1,'version':44,'save_path':'346802', 'released_date':'0010203'}))
# print(app.retrieve_app({'id':1}))
#
# remedy = RemedyMethod()
# print(remedy.retrieve_remedy_method_by_id('40'))
# save = {'rm_type': 'emotion', 'symptom': 'boring',
#         'provider': 'Healthline', 'url': 'https://www.healthline.com/health/boredom#prevention',
#         'maintain': '0', 'improve': '1', 'prevent': '0', 'description':
#             'Be prepared to take time out to work with your '
#             'child to set up an activity when they’re bored.',
#         'edit_date': '20180101', 'tag': ""}
# info = {'rm_type':'emotion1', 'symptom':'sad','provider':'Healthlines', 'url':'http://test',
#         'maintain':'1', 'improve':'0','prevent':'1', 'description':'test',
#         'edit_date': '20190101','tag':'test'}
# print(remedy.update_remedy_method(40, save))
# print(remedy.retrieve_remedy_method_by_id('40'))
#
#
#
#
# face_model = FaceRecognitionModel()
#
# print(type(face_model.retrieve_models(m_in)))
# m_in = {'start_date':'20190711'}
# print(pd.DataFrame(face_model.retrieve_models(m_in)))
# print(type(face_model.retrieve_models(m_in)))
# m_in = {'end_date':'20190711'}
# print(pd.DataFrame(face_model.retrieve_models(m_in)))
# print(type(face_model.retrieve_models(m_in)))
# m_in = {'start_date':'20180701', 'duration':'a month'}
# print(pd.DataFrame(face_model.retrieve_models(m_in)))
# print(type(face_model.retrieve_models(m_in)))
#
#
# m_in = {'id':131, 'staff_id':1}
# print(pd.DataFrame(face_model.retrieve_model_by_ids(m_in)))
# print(type(face_model.retrieve_model_by_ids(m_in)))
# m_in = {'training_set_id':45, 'staff_id':3}
# print(pd.DataFrame(face_model.retrieve_model_by_ids(m_in)))
# print(type(face_model.retrieve_model_by_ids(m_in)))
# m_in = {'staff_id':2}
# print(pd.DataFrame(face_model.retrieve_model_by_ids(m_in)))
# print(type(face_model.retrieve_model_by_ids(m_in)))
#
# print(face_model.retrieve_latest_model_id('aging_model'))
#
# app = AAAApp()
# app_in = {'version':1}
# print(pd.DataFrame(app.retrieve_app(app_in)))
# print(type(app.retrieve_app(app_in)))
# app_in = {'start_date':'20190705'}
# print(pd.DataFrame(app.retrieve_app(app_in)))
# print(type(app.retrieve_app(app_in)))
# app_in = {'end_date':'20190704'}
# print(pd.DataFrame(app.retrieve_app(app_in)))
# print(type(app.retrieve_app(app_in)))
#
# print(type(app.retrieve_latest_app_id()))
#
# remedy = RemedyMethod()
# print(pd.DataFrame(remedy.retrieve_all_rm('wrinkle')))
# print(type(remedy.retrieve_all_rm('wrinkle')))
#
# print(pd.DataFrame(remedy.retrieve_remedy_method_by_id(12)))
# print(type(remedy.retrieve_remedy_method_by_id(12)))
#
# provider = {'remedy_method_id':'15', 'provider':'NHS'}
# print(pd.DataFrame(remedy.retrieve_rm_by_provider(provider)))
# provider = {'provider':'WebMD', 'tag':'therapy'}
# print(pd.DataFrame(remedy.retrieve_rm_by_provider(provider)))
# provider = {'provider':'NHS'}
# print(pd.DataFrame(remedy.retrieve_rm_by_provider(provider)))
# provider = {'tag':'therapy'}
# print(pd.DataFrame(remedy.retrieve_rm_by_provider(provider)))

#
# for i in inputlist:
#     print('\n\n\nUser')
#     print(pd.DataFrame(user.retrieve_user(i)))
#     print(type(user.retrieve_user(i)))
#     # print('\n\n\nUserManager')
#     # print(pd.DataFrame(userm.retrieve_user(i)))
#     # print(type(userm.retrieve_user(i)))
#     print('\n\n\nStaff')
#     print(pd.DataFrame(staff.retrieve_staff(i)))
#     print(type(staff.retrieve_staff(i)))
#     # print('\n\n\nStaffManager')
#     # print(pd.DataFrame(staffm.retrieve_staff(i)))
#     # print(type(staffm.retrieve_staff(i)))
#



