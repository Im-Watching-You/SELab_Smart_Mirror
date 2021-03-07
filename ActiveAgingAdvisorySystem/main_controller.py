"""
Date: 2019.06
Programmer: MH
Description: Main Controller for AAA System
"""
import os
import platform
import shutil
from queue import Queue
from threading import Thread
import time

import cv2
import numpy as np
from gtts import gTTS
from time import sleep
import pyglet

from ActiveAgingAdvisorySystem.photo_manager import PhotoTaker
from ActiveAgingAdvisorySystem.recommender import RecommendationController
from profile_manager import ProfileManager, UserManager
from ActiveAgingAdvisorySystem.person import User, Staff
from ActiveAgingAdvisorySystem import factor
from ActiveAgingAdvisorySystem.photo import Photo
from ActiveAgingAdvisorySystem.aging_summary import AgingSummary
from ActiveAgingAdvisorySystem.session import Session
from ActiveAgingAdvisorySystem.aging_factors import AgingAppearance, AgingDiagnosis, AgingEvolution
from ActiveAgingAdvisorySystem.emotion_analytics import EmotionPrediction, EmotionDiagnosis, EmotionEvolutionAnalyzer
from ActiveAgingAdvisorySystem.predict_age import AgingDiagnosis as WinkleAD
from ActiveAgingAdvisorySystem.predict_spots import AgingDiagnosis as SpotAD
from ActiveAgingAdvisorySystem.progressanalyzer import EmotionProgressAnalyzer, AgingProgressAnalyzer
from ActiveAgingAdvisorySystem.train_emotion_classifier import FactorDetector
from ActiveAgingAdvisorySystem.aging_geometric import AgingGeoDetector
from ActiveAgingAdvisorySystem.factors import Factor


class MainController:
    def __init__(self):
        self.is_session = False
        self.pt = PhotoTaker()
        self.recm = RecommendationController()
        self.usermng = UserManager()
        self.db_user = User()
        self.summary = AgingSummary()
        self.db_sess = Session()
        self.db_photo = Photo()
        self.frame = None
        self.main_thread = False
        self.cur_user_info = {}
        self.user_id = 0
        self.f_name = None
        self.curr_pt_id = -1
        self.curr_sess_id = -1
        self.curr_detected_age_info = {}

        self.aa = AgingAppearance()
        self.ad = AgingDiagnosis()
        self.ae = AgingEvolution()

        self.ea = EmotionPrediction()
        self.ed = EmotionDiagnosis()
        self.ee = EmotionEvolutionAnalyzer()

        self.winkle_queue = Queue()
        self.spot_queue = Queue()
        self.geo_queue = Queue()
        self.emo_queue = Queue()

        self.epa = EmotionProgressAnalyzer()
        self.apa = AgingProgressAnalyzer()
        self.fs = Factor()
        self.running = True

        self.curr_ass_id = -1
        self.duration_progression = 8
        self.path_checker()

        self.curr_face = None
        self.curr_tag = -1

    def get_name(self, id):
        for i in self.db_user.get_id_list():
            if i["id"] == id:
                return i['first_name'], i['last_name']

    def run(self):
        """
        To run the system
        :return:
        """
        # self.save_thread = True
        pass

    def stop(self):
        """
        To stop the system (thread)
        :return:
        """
        self.main_thread = False
        self.save_thread = False

    def get_recm(self):
        result = []
        result_user = self.db_user.retrieve_user({'user_id': self.user_id})
        user = result_user[0]
        assessment = {'age': {'id': 514, 'assess_type': 'aging_diagnosis', 'm_id': 148, 'p_id': 1, 's_id': 619,
                              'result': 32, 'recorded_date': '2019-07-09 15:11:38'}, 'age_ft': [{'wrinkle': 32}]}
        # photo ={'id':self.curr_pt_id, 's_id':self.curr_sess_id}
        photo = {'id': 1, 's_id': 619}
        rm_list = self.recm.recom_rm(user, assessment, photo)
        for r in list(rm_list.values())[0]:
            result.append(">>>> " + r['description'])
            # result.append("")
        return result

    def set_user_info(self, dict_user):
        self.cur_user_info = dict_user

    def update_user_info(self, dict_user):
        self.usermng.update_user_profile(self.user_id, dict_user)
        self.get_cur_user_info(self.user_id)

    def set_session(self, s, id):
        """
        To set session
        :param s: bool, current session status
        :return:
        """
        print(">>> set_session: ", str(s), id)
        self.is_session = s
        if self.is_session:
            self.set_curr_user(id)
            self.db_sess.start_session(self.user_id)
            self.curr_sess_id = self.db_sess.retrieve_active_session_id(self.user_id)
            self.summary.set_curr_id(id)
            if id > 0:
                self.get_cur_user_info(id)
                if self.f_name is not None:
                    os.remove(self.f_name)
        else:
            self.set_curr_user(id)
            self.db_sess.finish_session(self.user_id)
            self.curr_sess_id = -1
            self.cur_user_info = {}
            self.summary.set_curr_id(id)

    def get_cur_user_info(self, id):
        self.cur_user_info = self.db_user.retrieve_user({"user_id": id})[0]

    def get_session(self):
        """
        To return current session
        :return: is_session, bool current session
        """
        return self.is_session

    def predict_age(self, img):
        self.aa.predict_age(img, data={"photo_id": self.curr_pt_id, "session_id": self.curr_sess_id})
        age = self.aa.get_result()
        self.curr_detected_age_info = self.aa.get_curr_predicted_age(
            data={"photo_id": self.curr_pt_id, "session_id": self.curr_sess_id})
        self.curr_ass_id = self.aa.curr_ass_id
        return age

    def predict_emotion(self, img):
        emotion = self.ea.predict_emotion(img, data={"photo_id": self.curr_pt_id, "session_id": self.curr_sess_id})
        self.curr_detected_emotion_info = self.ea.get_curr_predicted_emotion(
            data={"photo_id": self.curr_pt_id, "session_id": self.curr_sess_id})
        self.curr_ass_id = self.ea.curr_ass_id
        return emotion

    def set_curr_state(self, img, tag):
        self.curr_face = img
        self.curr_tag = tag

    def compute_age(self, curr_face):
        if self.geo_queue.qsize() <= 1:
            AgingGeoDetector(self.geo_queue, curr_face).start()
        if self.spot_queue.qsize() <= 1:
            SpotAD(self.spot_queue, curr_face).start()
        if self.winkle_queue.qsize() <= 1:
            WinkleAD(self.winkle_queue, curr_face).start()
        try:
            print("compute_age: ", self.winkle_queue.qsize(), self.spot_queue.qsize(), self.geo_queue.qsize())
        except:
            print("compute_age: Nothing")

    def compute_emotion(self, curr_face):
        FactorDetector(self.emo_queue, curr_face).start()
        try:
            print("compute_emotion: ", self.emo_queue.qsize())
        except:
            print("compute_emotion: Nothing")

    def predict_age_factor(self, img):
        try:
            age_apearance = self.predict_age(img)
            result_wrinkle = self.winkle_queue.get(0)
            result_spot = self.spot_queue.get(0)
            result_geo = self.geo_queue.get(0)
            if age_apearance is not None and result_wrinkle is not None and result_spot is not None and result_geo is not None:
                wk_age = result_wrinkle['Age']
                wrinkle = result_wrinkle['Wrinkle Features']
                dict_winkle = result_wrinkle["Whole Features"]
                result_wrinkle["Age_Wrinkle"] = result_wrinkle.pop("Age")

                spot = result_spot.pop('Aging_Spot_Features_Total')
                as_age = result_spot['Age']
                result_spot["Age_Spot"] = result_spot.pop("Age")

                obj_age, factors = result_geo['Age'], result_geo['Factor']
                factors['Age_Geo'] = factors.pop('age')
                dic = {**dict_winkle, **factors, **result_spot}
                if self.curr_ass_id >= 0:
                    self.fs.register_age_factor(self.curr_ass_id, dic)

                return {"Appearance": int(round(age_apearance)),
                        "Wrinkle": int(round(wk_age)),
                        "Aging Spot": int(round(as_age)),
                        "Distance (Glabella to Chin)": int(round(factors["Age_Geo"])),
                        "Shape (Area of Nose)": int(round(factors["Age_Geo"]))}
            else:
                return None
        except:
            return None

    def predict_emotion_factor(self):
        try:
            r = self.emo_queue.get(0)
            if r is None:
                result = None
            else:
                result, emotion_label, value_predicted = r
                result['emotion'] = value_predicted
                if self.curr_ass_id >= 0:
                    self.fs.register_emotion_factor(self.curr_ass_id, result)
            if result is not None:
                return {'Rate of Right Eye to Right Eyebrow': str(int(round(result['f1'], 2) * 100)) + "%",
                        'Rate of Left Eye to Left Eyebrow': str(int(round(result['f2'], 2) * 100)) + "%",
                        'Rate of Inner Ends of Eyebrows': str(int(round(result['f3'], 2) * 100)) + "%",
                        'Rate of Right Eye Corner to Right Mouth Corner': str(int(round(result['f4'], 2) * 100)) + "%",
                        'Rate of Left Eye Corner to Left Mouth Corner': str(int(round(result['f5'], 2) * 100)) + "%",
                        'Rate of Right Eyelid to Right Eyebrow': str(int(round(result['f6'], 2) * 100)) + "%",
                        'Rate of Left Eyelid to Left Eyebrow': str(int(round(result['f7'], 2) * 100)) + "%",
                        'Rate of Right Upper and Lower Eyelids': str(int(round(result['f8'], 2) * 100)) + "%",
                        'Rate of Left Upper and Lower Eyelids': str(int(round(result['f9'], 2) * 100)) + "%",
                        'Rate of Upper and Lower Lip': str(int(round(result['f10'], 2) * 100)) + "%",
                        'Rate of Outer Ends of Mouth': str(int(round(result['f11'], 2) * 100)) + "%"}
            else:
                return None
        except:
            return None

    def compute_emotion_progress(self):
        sve = self.epa.get_seasonal_variation_of_emotions(self.user_id, self.duration_progression)
        svf = self.epa.get_seasonal_variation_of_factors(self.user_id, self.duration_progression)
        return {'Emotion': sve, 'Factor': svf}

    def compute_aging_progression(self):
        sva = self.apa.get_seasonal_variation_of_age(self.user_id, self.duration_progression)
        svf = self.apa.get_seasonal_variation_of_factors(self.user_id, self.duration_progression)
        return {'Age': sva, 'Factor': svf}

    def get_age_charts(self, dic_duration, interval, dict_fact):
        c1, c2 = self.apa.get_trend_of_age(self.user_id,
                                           self.cur_user_info['first_name'] + " " + self.cur_user_info['last_name'],
                                           dic_duration, interval, dict_fact)
        return c1, c2

    def get_emotion_charts(self, dic_duration, interval, dict_fact):
        c1, c2 = self.epa.get_trend_of_emotions(self.user_id,
                                                self.cur_user_info['first_name'] + " " + self.cur_user_info[
                                                    'last_name'], dic_duration, interval, dict_fact)
        return c1, c2

    def get_session_id(self):
        self.db_sess.retrieve_active_session_id(self.user_id)

    def save_img(self, frame, user_id=0, mode="auto"):
        self.pt.save_img(frame, info={'user_id': self.user_id, 'session_id': self.curr_sess_id}, mode=mode)
        self.curr_pt_id = self.pt.get_curr_photo_info(self.curr_sess_id)

    def set_curr_user(self, u_id):
        self.user_id = u_id

    def get_curr_user(self):
        return self.user_id

    def register(self):
        pass

    def remove_img(self):
        pass

    def voice(self, content, file_name):
        self.f_name = ".\\tmp\\" + file_name + ".mp3"
        if not os.path.isfile(self.f_name):
            def th():
                try:
                    tts = gTTS(text=content, lang='en')
                    tts.save(self.f_name)
                    music = pyglet.media.load(self.f_name, streaming=False)
                    music.play()
                    sleep(music.duration)  # prevent from killing
                except:
                    pass

            Thread(target=th).start()
        else:
            pass

    def get_summaries(self):
        a_s, a_e = self.summary.aging_summary(7)
        e_s, e_e = self.summary.emotion_summary(7)
        return (a_s, a_e), (e_s, e_e)

    def copy_files(self, src, trg):
        if not os.path.exists(trg):
            os.mkdir(trg)
        for i in os.listdir(src):
            shutil.copy2(os.path.join(src, i), os.path.join(trg, i))

        for d in os.listdir(src):
            os.remove(os.path.join(src, d))

    def path_checker(self):
        """
        To check whether required path is or not.
        If a path isn't, create a path.
        :return: None
        """
        prereq_folder_list = ['./model', './models', './models/class', './models/aging_factor', './dataset',
                              './dataset/train', './dataset/test', './dummy', './models/aging_factor', './tmp',
                              './capture', './capture/automatic', './capture/manual', './saved_chart_image',
                              './dataset/rating/', './dataset/rating/implicit', './dataset/rating/explicit',
                              './model/cb', '../dataset', '../dataset/rm_emo_rating', './model/cf',
                              './models/emotion_factor', './resource', './FaceRecognition']

        if platform.system() == 'Windows':
            for i in range(0, len(prereq_folder_list)):
                count = 0
                for j in prereq_folder_list[i]:
                    if j == '/':
                        count += 1
                prereq_folder_list[i] = prereq_folder_list[i].replace('/', '\\', count)

        for path in prereq_folder_list:
            if not os.path.isdir(path):
                os.mkdir(path)


if __name__ == '__main__':
    mc = MainController()
    # mc.take_photo_auto()
    mc.run()
