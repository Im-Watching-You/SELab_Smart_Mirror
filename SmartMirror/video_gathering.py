"""
Date: 2019.05.27
Programmer: MH
Description: class for video gathering
"""
import copy
import datetime
import os
import shutil
import time
import schedule

import cv2

from db_connector import UserProfile
from detector import Detector
from events import Personal, Public, Assess, Suggest, Diary
from flags import Flags, ButtonFlag
from buttons import Button, BtnGroup, BtnGroupManager
from listview import ListView
from text_view import TextView
from threading import Thread
from photo_manager import PhotoTaker
from face_detections import FaceRecognizer, FaceDetector
from greetings import Greeting


class VideoGathering:
    def __init__(self):
        self.scale = 1
        self.input_video = 0  # 0: Real time Video, Location: String, path of video
        self.auth_zoom = False

        # self.WIDTH = 1600
        # self.HEIGHT = 900
        self.WIDTH = 640
        self.HEIGHT = 480

        self.center_x = self.WIDTH / 2
        self.center_y = self.HEIGHT / 2

        self.file_path = None
        self.image = None

        self.cap = cv2.VideoCapture(self.input_video)
        self.cap.set(3, self.WIDTH)
        self.cap.set(4, self.HEIGHT)

        self.is_mirroring = True    # Mirroring variable
        self.do_acquisition = True  # Data Acquisition
        self.curr_face_frame = None

        self.selected_x = -1
        self.selected_y = -1
        self.user_id = 0
        self.past_count = -1
        self.auto_timer = 10

        self.curr_name = ""
        self.past_name = ""
        self.detected_time = 0
        self.greeting_time = 0

        self.difference_counting = -1
        self.frame_face = []
        self.cand_users = []

        self.txt_info = []
        self.is_clicked = False

        self.face_detector = FaceDetector()
        self.face_recognizer = FaceRecognizer()
        self.photo_taker = PhotoTaker(self.face_recognizer)
        self.greeter = Greeting()

        self.evt_public = Public()
        self.evt_personal = Personal()
        self.evt_assess = Assess()
        self.evt_suggest = Suggest()
        self.evt_diary = Diary()

        # self.detector = Detector()

    def acquire(self):
        count = 0
        self.bgm = BtnGroupManager(width=self.WIDTH, height=self.HEIGHT)
        self.list_view = ListView()
        self.list_view.set_size(width=self.WIDTH, height=self.HEIGHT)
        self.txt_view = TextView()
        # schedule.every(self.auto_timer).seconds.do(self.auto_cap)

        while self.do_acquisition:
            schedule.run_pending()
            self.do_acquisition, frame = self.cap.read()  # To capture image
            if self.is_mirroring:
                frame = cv2.flip(frame, 1)  # To use Mirror mode
            self.image = frame.copy()

            self.frame = frame.copy()
            self.face_detector.detect(self.image)
            nrof_faces = self.face_detector.get_nrof_faces()
            if nrof_faces > 0:
                frame, face = self.face_detector.choose_face(frame, self.selected_x, self.selected_y)
                self.greeter.check_greet_case(nrof_faces)
                self.greeter.set_detected_num(nrof_faces)
                if not self.greeter.get_is_greet():
                    self.face_recognizer.detect(face)
                    self.greeter.greet(0)

            else:   #
                # print("Nobody")
                pass
            self.bgm.set_size(frame.shape[1], frame.shape[0])
            self.list_view.set_size(frame.shape[1], frame.shape[0])
            frame = self.bgm.draw_btn(frame)
            list_x, list_y = self.bgm.group_zoom.get_under_location()
            self.list_view.set_XY(list_x, list_y)
            self.set_action(frame)
            self.txt_view.put_text(frame, txt=self.txt_info, loc=self.bgm.loc)
            # if self.past_count <= 1 or self.in_session:
            #     if count % 10 == 0:  # detect person about every 0.33 second
            #         frame = self.detect(frame)  # To detect identity
            # else:
            #     frame = self.detect(frame)  # To detect identity

            self.display(frame)
            key = cv2.waitKey(1)
            if Flags.Btn_capture:
                # Capture the image manually and save in manual folder of the registerer user
                self.photo_taker.save_img(self.image, self.user_id, mode='manual')
                #save_img(self.image)
                Flags.Btn_capture = False
            if key == ord('q'):
                # To stop threads
                self.do_acquisition = False
                self.evt_public.off()
                self.evt_personal.off()
                self.evt_assess.off()
                self.evt_suggest.off()
                break

        count += 1
        if count >= 30:
            count = 0

    # Capture the image automatically and save image following automatic saving process
    def auto_cap(self):
        if self.image is not None:
            print("Capturing!")
            Thread(target=self.photo_taker.save_img, args=(self.image, self.user_id)).start()

    def set_mirroring(self, is_mirroring):
        self.is_mirroring = is_mirroring

    def display(self, frame):
        cv2.setMouseCallback("Smart Mirror System", self.mouse_callback)
        # cv2.namedWindow("Smart Mirror System", cv2.WINDOW_NORMAL)
        # cv2.setWindowProperty("Smart Mirror System", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Smart Mirror System", frame)  # display image

    def mouse_callback(self, event, x, y, flag, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selected_x = x
            self.selected_y = y
            if Flags.Diary and ButtonFlag.diary_read_flag and self.list_view.in_event(x, y):
                self.list_view.detail_event(x, y)
            if Flags.Diary and ButtonFlag.diary_read_flag and self.list_view.in_list(x, y):
                self.list_view.listview_event(x, y)
            else:
                self.bgm.on_click(x, y)
        elif event == cv2.EVENT_LBUTTONDBLCLK:
            self.selected_x = x
            self.selected_y = y

    def set_action(self, frame):
        self.txt_info = []
        if Flags.Public:
            self.evt_public.on()
            self.txt_info.append(self.evt_public.time_info)
        else:
            self.evt_public.off()
        if Flags.Personal:
            self.evt_personal.on(self.user_id)
            self.txt_info.append(self.evt_personal.info)
        else:
            self.evt_personal.off()
        if Flags.Assess:
            self.evt_assess.on(self.frame_face, self.user_id)
        if Flags.Diary:
            self.evt_diary.on()
            if ButtonFlag.diary_read_flag:
                diary_data = self.evt_diary.event_reading()
                if diary_data is not None:
                    print('Diary-Read')
                    self.list_view.set_data(diary_data)
                    self.list_view.make_listview(frame)
            if ButtonFlag.record_flag:
                self.evt_diary.event_recording()

    def detect(self, frame):
        if len(self.detector.cand_id) != len(self.cand_users):
            self.load_cand()
            self.in_session = False
            self.is_greeting = False

        self.detector.detect_people(frame)  # To count People
        count = self.detector.get_nrof_faces()
        if count is None:  # To detect error
            return frame

        if self.past_count != count:  # To check name or id
            if self.in_session:
                self.in_session = False
                self.is_greeting = False
                self.is_clicked = False
                self.selected_x = -1
                self.selected_y = -1
                self.difference_counting = 0

        if count >= 1:  # If # of people is same or more than 1
            if count > 1:
                if not self.is_clicked:
                    frame, detected_face = self.detector.choose_face(frame, (self.selected_x*0.4),
                                                                     (self.selected_y*0.4))  # Tag Faces
                    # TODO: MH, clicked location will be changed.
                    if detected_face is not None:  # if face is detected well
                        self.curr_face_frame = detected_face

                        # save image from identified user for model training
                        #save_img(self.curr_face_frame, self.user_id, mode='detection')
                        idx, id = self.detector.identify_face(frame, detected_face)  # Recognize Face
                        if type(id) is not int:
                            return frame

                        if 0 < idx < len(self.cand_users):
                            self.user_id = id
                            self.curr_name = self.cand_users[idx]
                        else:
                            self.user_id = id
                            self.curr_name = "Unknown"

                        # self.curr_name, self.curr_face_frame = self.detector.get_identity_result()
                        if self.curr_name is None or self.curr_face_frame is None:
                            return frame
                        self.is_clicked = True
                        if not self.in_session:
                            self.in_session = True

            elif count == 1:  # if only one person is detected.
                # To set selected user information and tag
                frame, self.curr_face_frame = self.detector.choose_face(frame, None, None)  # Tag Faces
                #save_img(self.curr_face_frame, self.user_id, mode='detection')
                idx, id = self.detector.identify_face(frame, self.curr_face_frame)  # Recognize Face
                if type(id) is not int:
                    return frame
                if 0 < idx < len(self.cand_users):
                    self.user_id = id
                    self.curr_name = self.cand_users[idx]
                else:
                    self.user_id = id
                    self.curr_name = "Unknown"
                # self.curr_name, self.curr_face_frame = self.detector.get_identity_result()
                if self.curr_name is None or self.curr_face_frame is None:
                    return frame
                if not self.in_session:
                    self.in_session = True

            # To check registered
            if (self.difference_counting > 3 or self.difference_counting == -1) and not self.is_greeting:
                if self.in_session:
                    self.auth_zoom = True
                    if self.curr_name != "Unknown":  # Registered
                        self.registered = True
                        Flags.Unknown = False
                        if self.curr_name != self.past_name:  # To check name or id
                            if self.in_session:
                                self.in_session = False
                                self.is_greeting = False
                                self.is_clicked = False
                                # if not self.is_greeting:
                                #print(">>>", self.past_name, self.curr_name)
                                frame = self.greeting(frame, self.curr_name)
                            self.in_session = True
                        # To store Image
                    else:  # Not Registered
                        self.registered = False
                        self.curr_name = "Unknown"
                        Flags.Unknown = True
                        shutil.rmtree(".\\image\\0\\")
                        os.mkdir(".\\image\\0\\")
                        if not self.is_greeting:
                            frame = self.greeting(frame, self.curr_name)
                        #######################################################

            if self.curr_name != self.past_name or self.past_count != count:
                self.difference_counting += 1
            else:
                self.difference_counting = 0

        elif count == 0:  # If # of people is zero
            if self.in_session and self.is_greeting:  # if in session
                self.in_session = False  # close session
                self.is_clicked = False
                self.is_greeting = False
                self.registered = False
                self.user_id = -1000
                Flags.Unknown = True
                self.curr_name = ""
                self.past_name = ""
                self.difference_counting = 0
                self.selected_x, self.selected_y = -1, -1
        else:
            print("ERROR: Wrong Count")
        self.past_count = count
        # self.save_img(frame)
        #self.save_img(self.curr_face_frame)
        # frame = cv2.resize(frame, dsize=(0, 0), fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        return frame

    def greeting(self, frame, name):
        if "" != name and name is not None:
            if name == 'Unknown':
                self.greeter = 'Welcome to Smart Mirror!'
            else:
                self.greeter = "Hi! " + str(name)
            self.detector.voice(self.greeter)
            self.greeting_time = time.time()
            self.is_greeting = True
            self.past_name = self.curr_name
        return frame

    def load_cand(self):
        up = UserProfile()
        up.load_user_list_db()
        self.cand_users = []
        for u in up.list_user:
            #print(u.read_first_name())
            self.cand_users.append(u.read_first_name())


if __name__ == '__main__':
    vg = VideoGathering()
    vg.acquire()

