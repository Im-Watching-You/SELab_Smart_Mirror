# -*- coding: utf-8 -*-
"""
Date: 2019.06.
Programmer: MH
Description: User interface code for AAA System client tier
"""
import time
import tkinter as tk
import tkinter.ttk
from tkinter import Label, PhotoImage, Button, Entry
import cv2
from PIL import Image, ImageTk
import numpy as np
import sys
# sys.path.append(".\\ActiveAgingAdvisorySystem")
from ActiveAgingAdvisorySystem.recommender import RecommendationController
from ActiveAgingAdvisorySystem.main_controller import MainController
from ActiveAgingAdvisorySystem.aging_factors import AgingAppearance, AgingDiagnosis, AgingEvolution
from ActiveAgingAdvisorySystem.emotion_analytics import EmotionPrediction, EmotionDiagnosis, EmotionEvolutionAnalyzer
from ActiveAgingAdvisorySystem.face_detections import FaceDetector, FaceRecognizer
from ActiveAgingAdvisorySystem.virtual_keyboard import Keyboard
from datetime import datetime
from datetime import timedelta

# sys.path.append(".\\JC")
COLOR = ["#efb5ae", "#c8e7a7", "#7fd7f7", "#CCCCCC"]
SCALE = [1.0, 0.8, 0.6, 0.4, 0.2]


class UserInterface:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.btn_state = {"Aging": False, "Emotion": False, "Recommend": False}
        self.width, self.height = int(self.cap.get(3) * 1.4), int(self.cap.get(4) * 1.4)
        # self.width, self.height = int(self.cap.get(3)), int(self.cap.get(4))
        self.curr_color = COLOR[3]
        self.root = tk.Tk()
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        # self.screen_width = 896
        # self.screen_height = 672
        # print(screen_width, screen_height)
        # self.root.overrideredirect(1)
        # self.root.attributes("-fullscreen", True)     # Full Screen
        # self.root.attributes("")
        # self.width = screen_width
        # self.height = screen_height
        self.set_widow_size(self.width, self.height)
        self.root.bind('<Escape>', lambda e: self.root.quit())
        self.root.bind('<Button-1>', self.cbk_click)
        self.lmain = tk.Label(self.root)
        self.lmain.pack()
        self.lmain.bind('<Double-Button-1>', self.cbk_zoom_reset)
        self.start_time = time.time()

        self.mc = MainController()
        self.photo_taker = self.mc.pt
        self.is_session = False
        self.user_id = 0
        self.frame = None
        self.db_user = self.mc.usermng

        self.selected_x, self.selected_y = None, None
        self.curr_user_id = None
        self.curr_face = None

        self.recm = RecommendationController()

        self.ep = EmotionPrediction()
        self.ed = EmotionDiagnosis()
        self.ee = EmotionEvolutionAnalyzer()

        self.fd = FaceDetector()
        self.fr = FaceRecognizer()
        self.greeting_duration = 5
        self.curr_scale = 0
        self.last_num = 0

        self.last_greeting_time = 0
        self.disappeared_time = 0
        self.is_said = False
        self.do_draw_age = False
        self.window_keyboard = None
        self.is_opened_oth_pg = False
        self.skip_count = 0
        self.is_skipped = True

        self.win_progress = None
        self.info_view = None
        self.txt = None
        self.btn_anal = None
        self.btn_recm = None
        self.btn_evol = None

        self.open_win_setting = False
        self.open_win_keyboard = False
        self.open_win_progress = False

    def set_widow_size(self, w, h):
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.root.title("Active Aging Advisory System")

        # self.root.geometry(str(w)+"x"+str(h)+"+0+0")
        # self.root.resizable(False, False)

    def run(self):
        self.mc.run()
        self.draw_info_view()
        self.draw_btn_fnc()
        self.draw_frames()
        scale_x = int(self.screen_width / 896)
        scale_y = int(self.screen_height / 672)
        if scale_x > scale_y:
            scale_x = scale_y
        img_plus = PhotoImage(file='./resource/img_plus.png')
        img_plus = img_plus.zoom(scale_x)
        img_minus = PhotoImage(file='./resource/img_minus.png')
        img_minus = img_minus.zoom(scale_x)
        img_camera = PhotoImage(file='./resource/img_camera.png')
        img_camera = img_camera.zoom(scale_x)
        img_setting = PhotoImage(file='./resource/img_setting.png')
        img_setting = img_setting.zoom(scale_x)
        self.age_thread()
        self.btn_plus = Button(self.root, command=self.cbk_plus, activebackground="#123345", bg=self.curr_color,
                               borderwidth=0, image=img_plus)
        self.btn_minus = Button(self.root, command=self.cbk_minus, activebackground="#123345",
                                bg=self.curr_color, borderwidth=0, image=img_minus)
        self.btn_capture = Button(self.root, command=lambda: self.cbk_capture(self.user_id), activebackground="#123345",
                                  bg=self.curr_color, borderwidth=0, image=img_camera)
        self.btn_setting = Button(self.root, command=self.cbk_setting, activebackground="#123345",
                                  bg=self.curr_color, borderwidth=0, image=img_setting)
        self.btn_plus.place(rely=0.01, relx=0.83)
        self.btn_minus.place(rely=0.01, relx=0.87)
        self.btn_capture.place(rely=0.01, relx=0.91)
        self.btn_setting.place(rely=0.01, relx=0.95)

        self.root.mainloop()

    def age_thread(self):
        if self.is_session and not self.is_opened_oth_pg:
            if self.curr_face is not None and self.btn_state["Aging"]:
                self.mc.compute_age(self.curr_face)
            if self.curr_face is not None and self.btn_state["Emotion"]:
                self.mc.compute_emotion(self.curr_face)
            print("age_thread: hihi", datetime.now())
        self.root.after(2000, self.age_thread)

    def draw_btn_fnc(self):
        if self.btn_anal is None and self.btn_recm is None and self.btn_evol is None:
            self.btn_anal = Button(self.root, text="Aging Analytics", command=lambda: self.cbk_txt("Aging"),
                                   activebackground="#123345",
                                   borderwidth=0, bg=self.curr_color, bd=1)
            self.btn_recm = Button(self.root, text="Emotion Analytics", command=lambda: self.cbk_txt("Emotion"),
                                   activebackground="#123345",
                                   borderwidth=0, bg=self.curr_color, bd=1)
            self.btn_evol = Button(self.root, text="Recommend", command=lambda: self.cbk_txt("Recommend"),
                                   activebackground="#123345",
                                   borderwidth=0, bg=self.curr_color, bd=1)
        if self.btn_anal is not None and self.btn_recm is not None and self.btn_evol is not None:
            self.btn_anal.place(anchor="nw", relx=0.83, rely=0.07, width=int(130 * self.screen_width / 896),
                                height=int(40 * self.screen_height / 672))
            self.btn_recm.place(anchor="nw", relx=0.83, rely=0.15, width=int(130 * self.screen_width / 896),
                                height=int(40 * self.screen_height / 672))
            self.btn_evol.place(anchor="nw", relx=0.83, rely=0.23, width=int(130 * self.screen_width / 896),
                                height=int(40 * self.screen_height / 672))

    def remove_btn_fnc(self):
        self.btn_anal.place_forget()
        self.btn_recm.place_forget()
        self.btn_evol.place_forget()

    def draw_info_view(self):
        if self.info_view is None and self.txt is None:
            self.info_view = tk.Frame(self.root, bd=3, width=0, heigh=0)
            font = tkinter.font.Font(size=11, weight='bold')
            self.txt = tk.Text(self.info_view, width=40, heigh=30, wrap="word", borderwidth=0, highlightthickness=0,
                               relief="flat",
                               background="#EEEEEE", exportselection=False, cursor="arrow", insertwidth=0, font=font)
            self.txt.bind("<Button-1>", self.cbk_info_view)
        self.info_view.place(relx=0.01, rely=0.05)

    def remove_info_view(self):
        self.info_view.place_forget()
        self.txt.pack_forget()

    def set_txt(self, txt):
        if txt is None:
            pass

        else:
            if type(txt) == str:
                self.txt.delete(1.0, tk.END)
                self.txt.insert("current", txt + "\n")
            else:
                self.txt.delete(1.0, tk.END)
                for t in txt:
                    self.txt.insert("current", t + "\n")

    def get_curr_frame(self):
        return self.frame

    def draw_progress_view(self, type):
        self.win_progress = tk.Toplevel()
        self.win_progress.geometry("1100x650+20+20")
        self.win_progress.overrideredirect(1)
        self.win_progress.bind("<Destroy>", self._destroy_progress)
        # 엔트리 배치
        self.ent_info_view()
        # 체크박스 배치
        self.chk_info_view(type)
        # 레이블 배치
        self.lbl_info_view()
        # 버튼 배치
        self.btn_info_view(type)

        # self.open_win_progress = True
        self.is_opened_oth_pg = True

    def lbl_info_view(self):

        big_font = tk.font.Font(size=20, weight='bold')
        small_font = tk.font.Font(size=11)

        l_options = Label(self.win_progress, text="Options", justify="left", font=big_font)
        l_options.place(relx=0.02, rely=0.01)

        l_wave = Label(self.win_progress, text="~", justify="left", font=big_font)
        l_wave.place(relx=0.15, rely=0.08)

        l_date = Label(self.win_progress, text="Date: ", justify="left", font=small_font)
        l_date.place(relx=0.03, rely=0.08)

        l_legend = Label(self.win_progress, text="Legends: ", justify="left", font=small_font)
        l_legend.place(relx=0.03, rely=0.13)

        l_interval = Label(self.win_progress, text="Interval: ", justify="left", font=small_font)
        l_interval.place(relx=0.03, rely=0.18)

        self.combo_str = tk.StringVar()
        self.combo_interval = tk.ttk.Combobox(self.win_progress, width=20, textvariable=self.combo_str)
        self.combo_interval['values'] = ('Daily', 'Weekly', 'Monthly', 'Quarterly', 'Yearly')
        self.combo_interval.place(relx=0.09, rely=0.18, width=80, height=30)

        l_graph_spec = Label(self.win_progress, text="Progression", justify="left", font=big_font)
        l_graph_spec.place(relx=0.02, rely=0.25)

    def ent_info_view(self):

        # Set default value of date
        default_start_date = datetime.strftime(datetime.now() - timedelta(days=6), '%Y-%m-%d')
        default_end_date = datetime.strftime(datetime.now(), '%Y-%m-%d')

        self.start_entry = Entry(self.win_progress)
        self.end_entry = Entry(self.win_progress)

        # insert value at [0]
        self.start_entry.insert(0, default_start_date)
        self.end_entry.insert(0, default_end_date)

        self.start_entry.place(relx=0.09, rely=0.08, width=70, height=30)
        self.end_entry.place(relx=0.17, rely=0.08, width=70, height=30)

    def chk_info_view(self, graph_type):
        """
        To put check button on the Window
        :param graph_type: String, one of ['Aging', 'Emotion'], Type of graph determines what to put on the Window
        :return:
        """
        self.age_factor_list = ['Age', 'Age_Wrinkle', 'Age_Spot', 'Age_Geo', 'Appearance']
        self.emotion_factor_list = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11']
        self.chkbt_variety_list = []

        if graph_type == "Aging":
            chkbt_list = []
            num_of_factor = 5

            # To set loc of check buttons
            x_loc = [0.07 + 0.075*i for i in range(5)]
            y_loc = 0.13

            for i in range(num_of_factor):
                self.chkbt_variety_list.append(tk.IntVar())
                chkbt_list.append(tk.Checkbutton(self.win_progress, text=self.age_factor_list[i],
                                                 variable=self.chkbt_variety_list[i]))
                chkbt_list[i].place(relx=x_loc[i], rely=y_loc, width=100, height=30)

        elif graph_type == 'Emotion':
            chkbt_list = []
            num_of_factor = 11

            # To set loc of check buttons
            x_loc = [0.07 + 0.061*i for i in range(11)]
            y_loc = 0.13

            for i in range(num_of_factor):
                self.chkbt_variety_list.append(tk.IntVar())
                chkbt_list.append(tk.Checkbutton(self.win_progress, text=self.emotion_factor_list[i],
                                                 variable=self.chkbt_variety_list[i]))
                chkbt_list[i].place(relx=x_loc[i], rely=y_loc, width=100, height=30)

    def btn_info_view(self, graph_type):
        self.btn_day = Button(self.win_progress, text="Accept",
                               command=lambda: self.cbk_display_graph(graph_type, interval=self.combo_str.get(),
                                                                      duration=[self.start_entry.get(), self.end_entry.get()]),
                               activebackground="#123345",
                               borderwidth=0, bg=self.curr_color, bd=1)

        self.btn_day.place(relx=0.19, rely=0.18, width=80, height=30)

    def cbk_display_graph(self, graph_type, interval, duration):
        """
        :param graph_type: String, 'Aging' or 'Emotion'
        :param interval: String, one of ['day', 'week', 'month', 'quarter', 'year']
        :param duration: List, [ ((str)start_date), ((str)end_date)
        """

        dict_duration = {'start_date': duration[0], 'end_date': duration[1]}

        # To set a empty list to fill in String of check button type
        button_state = []

        if graph_type == 'Aging':
            # To check each check button's state (1: checked, 0: non-checked)
            for i in range(len(self.chkbt_variety_list)):
                if self.chkbt_variety_list[i].get() == 1:
                    button_state.append(self.age_factor_list[i])

        elif graph_type == 'Emotion':
            # To check each check button's state (1: checked, 0: non-checked)
            for i in range(len(self.chkbt_variety_list)):
                if self.chkbt_variety_list[i].get() == 1:
                    button_state.append(self.emotion_factor_list[i])

        if button_state is []:
            print('Check button Empty.')
            return
        else:
            # print(button_state)
            dict_fact = {'factors': button_state}

        # To check type of graph
        if graph_type == 'Aging':
            # hist, pie = self.mc.get_age_charts()
            hist, pie = self.mc.get_age_charts(dict_duration, interval, dict_fact)

        elif graph_type == 'Emotion':
            # hist, pie = self.mc.get_emotion_charts()
            hist, pie = self.mc.get_emotion_charts(dict_duration, interval, dict_fact)
        else:
            print('Check \'graph_type\' value. It\'s invalid.')
            return

        img1 = Image.open(hist)

        # To get size of original image
        width, height = img1.size

        if graph_type == 'Aging':
            # Resize the image by ratio
            img1 = img1.resize((int(width*0.8), int(height*0.8)), Image.ANTIALIAS)
        elif graph_type == 'Emotion':
            img1 = img1.resize((int(width*0.65), int(height*0.7)), Image.ANTIALIAS)
        # To put image into window
        img1 = ImageTk.PhotoImage(image=img1)
        label = Label(self.win_progress, image=img1)
        label.image = img1
        label.place(relx=0.01, rely=0.35)

        img2 = Image.open(pie)
        # To get size of original image
        width, height = img2.size

        # Resize the image by ratio
        img2 = img2.resize((int(width*0.7), int(height*0.7)), Image.ANTIALIAS)

        # To put image into window
        img2 = ImageTk.PhotoImage(image=img2,  width=70, height=70)

        label = Label(self.win_progress, image=img2)

        label.image = img2
        if graph_type == 'Aging':
            label.place(relx=0.51, rely=0.35)
        elif graph_type == 'Emotion':
            label.place(relx=0.57, rely=0.35)

    def draw_win_feedback(self, date="2019.06.28", tag="Emotion"):
        frame_feedback = tk.Tk()
        frame_feedback.geometry("320x320+100+100")
        frame_feedback.resizable(False, False)
        frame_feedback.overrideredirect(1)

        l_date = Label(frame_feedback, text=date, justify="left")
        l_date.grid(column=10, row=0, columnspan=7)
        l_tag = Label(frame_feedback, text="[Emotion]", justify="left")
        l_tag.grid(column=0, row=1, columnspan=10, sticky='w')
        l_detected = Label(frame_feedback, text="Detected Emotion: Sad", justify="left")
        l_detected.grid(column=0, row=2, columnspan=10, sticky='w')
        l_remdy = Label(frame_feedback, text="Listen classical music", justify="left")
        l_remdy.grid(column=0, row=3, columnspan=10, sticky='w')

        l_rating = Label(frame_feedback, text="Rating:", justify="left")
        l_rating.grid(column=0, row=5, columnspan=10, sticky='w')

        btn_1 = Button(frame_feedback, text="1", command=self.cbk_signup,
                       bg=self.curr_color, bd=1, width=4, height=2)
        btn_1.grid(column=1, row=6, columnspan=10, sticky='w')
        btn_2 = Button(frame_feedback, text="2", command=self.cbk_signup,
                       bg=self.curr_color, bd=1, width=4, height=2)
        btn_2.grid(column=4, row=6, columnspan=10, sticky='w')
        btn_3 = Button(frame_feedback, text="3", command=self.cbk_signup,
                       bg=self.curr_color, bd=1, width=4, height=2)
        btn_3.grid(column=7, row=6, columnspan=10, sticky='w')
        btn_4 = Button(frame_feedback, text="4", command=self.cbk_signup,
                       bg=self.curr_color, bd=1, width=4, height=2)
        btn_4.grid(column=10, row=6, columnspan=10, sticky='w')
        btn_5 = Button(frame_feedback, text="5", command=self.cbk_signup,
                       bg=self.curr_color, bd=1, width=4, height=2)
        btn_5.grid(column=16, row=6, columnspan=10, sticky='w')

    def draw_frames(self):
        """
        Main Thread
        :return: None
        """
        # if self.frame is not None:
        _, self.frame = self.cap.read()
        self.frame = cv2.flip(self.frame, 1)
        if not self.is_opened_oth_pg:
            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            # if time.time() - self.start_time > 0.01:
            self.fd.detect(gray)
            self.frame, self.curr_face = self.fd.choose_face(self.frame, self.selected_x, self.selected_y)
            if self.fd.nrof_faces == 1:
                self.selected_y, self.selected_x = None, None
                self.fr.nrof_faces = self.fd.nrof_faces
                dtt_usr_id = self.fr.recognize(self.curr_face)
                self._check_skip(dtt_usr_id)
                self.curr_user_id = dtt_usr_id
                # self.curr_user_id = (2, 2)
                self.last_num = self.fd.nrof_faces
                if not self.is_skipped:
                    if self.last_greeting_time == 0:
                        self.last_greeting_time = time.time()
                    # if int(self.curr_user_id[0]) > 0:
                    if not self.is_session:  # Add Time
                        if time.time() - self.last_greeting_time < self.greeting_duration:
                            self.show_greeting()
                        else:
                            self.mc.set_session(True, int(self.curr_user_id[0]))
                            self.is_session = True
                            self.is_skipped = True
                            self.mc.save_img(self.curr_face, mode="face")
                            self.is_said = False
                            # self.age_thread()
                            self.mc.set_curr_state(self.curr_face, -1)
                else:
                    # self.age_thread()
                    if not self.btn_state["Aging"] and not self.btn_state['Emotion'] and not self.btn_state[
                        "Recommend"]:
                        self.show_summary()
                        self.mc.set_curr_state(self.curr_face, -1)
                    else:
                        if self.btn_state['Aging']:
                            self.mc.set_curr_state(self.curr_face, 0)
                        elif self.btn_state['Emotion']:
                            self.mc.set_curr_state(self.curr_face, 1)
                        # self.age_thread()
                        self.gather_data()
                    self.mc.save_img(self.frame, mode="auto")
                    self.mc.save_img(self.curr_face, mode="face")
                    self.disappeared_time = time.time()

            elif self.fd.nrof_faces > 1:
                self.info_view.place(relx=0.01, rely=0.05)
                self.fr.nrof_faces = self.fd.nrof_faces
                if self.selected_x is not None and self.selected_y is not None:
                    self.fd.choose_face(self.frame, self.selected_x, self.selected_y)
                    dtt_usr_id = self.fr.recognize(self.curr_face)
                    self._check_skip(dtt_usr_id)
                    self.curr_user_id = dtt_usr_id
                    # self.curr_user_id = (2, 2)
                    self.last_num = self.fd.nrof_faces
                    if not self.is_skipped:
                        if self.last_greeting_time == 0:
                            self.last_greeting_time = time.time()
                        if not self.is_session:  # Add Time
                            if time.time() - self.last_greeting_time < self.greeting_duration:
                                self.show_greeting()
                            else:
                                self.mc.set_session(True, int(self.curr_user_id[0]))
                                self.is_session = True
                                self.is_skipped = True
                                self.mc.save_img(self.curr_face, mode="face")
                                self.is_said = False
                                self.mc.set_curr_state(self.curr_face, -1)
                                # self.age_thread()
                    else:
                        # self.age_thread()
                        if not self.btn_state["Aging"] and not self.btn_state['Emotion'] and not self.btn_state[
                            "Recommend"]:
                            self.show_summary()
                            self.mc.set_curr_state(self.curr_face, -1)
                        else:
                            if self.btn_state['Aging']:
                                self.mc.set_curr_state(self.curr_face, 0)
                            elif self.btn_state['Emotion']:
                                self.mc.set_curr_state(self.curr_face, 1)
                            self.gather_data()
                        self.mc.save_img(self.frame, mode="auto")
                        self.mc.save_img(self.curr_face, mode="face")
                        self.disappeared_time = time.time()

            elif self.fd.nrof_faces == 0 or self.curr_face is None:
                if time.time() - self.disappeared_time > self.greeting_duration * 2:
                    self._reset_session()
                # self.start_time = time.time()
                # self.show_summary()
        else:
            # self.age_thread()
            if not self.btn_state["Aging"] and not self.btn_state['Emotion'] and not self.btn_state["Recommend"]:
                self.show_summary()
                self.mc.set_curr_state(self.curr_face, -1)
            else:
                if self.btn_state['Aging']:
                    self.mc.set_curr_state(self.curr_face, 0)
                elif self.btn_state['Emotion']:
                    self.mc.set_curr_state(self.curr_face, 1)
                self.gather_data()
            self.mc.save_img(self.frame, mode="auto")
            self.mc.save_img(self.curr_face, mode="face")
            self.disappeared_time = time.time()
        x, y = self.width, self.height
        s = (1 - SCALE[self.curr_scale]) / 2
        crop = self.frame[int(y * s):y - int(y * s), int(x * s):x - int(x * s)]
        y = self.screen_height
        x = int(16 * y / 9)
        frame = cv2.resize(crop, (x, y), interpolation=cv2.INTER_CUBIC)

        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.lmain.imgtk = imgtk
        self.lmain.configure(image=imgtk)
        self.lmain.after(10, self.draw_frames)  # to make thread repeated show_frames method every 10 ms

    def show_greeting(self):
        self.draw_btn_fnc()
        self.draw_info_view()
        self.txt.delete(1.0, tk.END)
        f_name, l_name, g1 = "", "", ""
        if int(self.curr_user_id[0]) > 0:
            f_name, l_name = self.mc.get_name(int(self.curr_user_id[0]))
            g1 = "Hi! " + f_name + " " + l_name + "."
            self.txt.insert("current", g1 + " \n")
        g2 = "Welcome to Active Aging Advisory."
        self.txt.insert("current", g2)
        self.txt.pack()
        if not self.is_said:
            self.mc.voice(g1+g2, "greeting_"+f_name+"_"+l_name)
            self.is_said = True

    def show_summary(self):
        self.txt.delete(1.0, tk.END)
        name = ""
        if self.is_session and int(self.curr_user_id[0]) > 0 and self.mc.cur_user_info != {}:
            name = self.mc.cur_user_info["first_name"]+", "
        self.txt.insert("current", name + "Check aging information.\n")
        self.txt.insert("current", "\n")
        self.txt.insert("current", "Today is " + self.mc.summary.weather + ".\n")
        self.txt.insert("current", "    Temperature: " + str(self.mc.summary.temper) + u'\u2103' + "\n")
        self.txt.insert("current", "    Humidity: " + str(self.mc.summary.humid) + "%\n")
        self.txt.insert("current", "    UV Index: " + str(self.mc.summary.uvi) + "\n")
        self.txt.insert("current", "\n")
        if self.is_session and int(self.curr_user_id[0]) > 0:
            (a_s, a_e), (e_s, e_e) = self.mc.get_summaries()
            aging = ""
            emotion = ""
            diff_age = 0
            if a_s > 0 and a_e > 0:
                diff_age = a_e - a_s
                if diff_age > 0:
                    aging = "You look older"
                elif diff_age < 0:
                    aging = "You look younger"
                else:
                    aging = "You maintain aging"
            if e_s != "" and e_e != "":
                emotion = " and " + e_e + " during an week."
            if aging != "" and emotion != "":
                self.txt.insert("current", aging + emotion + "\n")
            if a_s > 0 and a_e > 0:
                a_s = int(round(a_s))
                a_e = int(round(a_e))

                if a_s - a_e != 0:
                    self.txt.insert("current", "    Aging (Age for an week): " + str(round(diff_age)) +"("+str(a_s)+
                                    u"\u2192"+str(a_e)+")\n")
                else:
                    self.txt.insert("current", "    Aging (Age for an week): " + str(round(diff_age)) + "(" + str(a_s) +" Maintaining)\n")
            else:
                self.txt.insert("current", "    Aging (Age for an week): No Data. \n")
            if e_s != "" and e_e != "":
                if e_s != e_e:
                    self.txt.insert("current", "    Emotion: " + str(e_s).capitalize() + " " + u"\u2192" + " " + str(
                        e_e).capitalize() + "\n")
                else:
                    self.txt.insert("current", "    Emotion: " + str(e_s).capitalize() + " (Maintaining)" + "\n")
            else:
                self.txt.insert("current", "    Emotion: No Data. \n")
            self.txt.insert("current", "\n")
        self.txt.insert("current", self.mc.summary.get_summary_tips() + "\n")
        self.txt.pack()

    def get_current_session(self, sess, id):
        self.is_session = sess
        self.user_id = id

    def cbk_setting(self):
        self.win_setting = tk.Tk()
        self.win_setting.geometry("320x320+200+200")
        # self.win_setting.resizable(False, False)
        self.win_setting.overrideredirect(1)
        self.win_setting.bind("<Destroy>", self._destroy_setting)
        self.is_opened_oth_pg = True
        self.open_win_setting = True
        notebook = tk.ttk.Notebook(self.win_setting, width=300, height=300)
        notebook.pack()

        frame2 = tk.Frame(self.win_setting)
        if not self.mc.is_session:
            notebook.add(frame2, text="Register")
        else:
            notebook.add(frame2, text="User Info")
        self.dict_user = {"first_name": "", "last_name": "", "email": "", "password": "", "gender": "",
                          "birth_date": "", "phone_number": ""}
        input_fir_name = ""
        l_text = Label(frame2, text="First Name", justify="left")
        l_text.grid(column=0, row=0)
        self.entry_fir_name = Entry(frame2, width=20, textvariable=input_fir_name)
        self.entry_fir_name.bind("<Button-1>", lambda event, obj=self.entry_fir_name: self.cbk_keyboard(event, obj))
        self.entry_fir_name.grid(column=1, row=0)
        input_last_name = ""
        l_text = Label(frame2, text="Last Name", justify="left")
        l_text.grid(column=0, row=1)
        self.entry_last_name = Entry(frame2, width=20, textvariable=input_last_name)
        self.entry_last_name.bind("<Button-1>", lambda event, obj=self.entry_last_name: self.cbk_keyboard(event, obj))
        self.entry_last_name.grid(column=1, row=1)
        input_email = ""
        l_text = Label(frame2, text="Email", justify="left")
        l_text.grid(column=0, row=2)
        self.entry_email = Entry(frame2, width=20, textvariable=input_email)
        self.entry_email.bind("<Button-1>", lambda event, obj=self.entry_email: self.cbk_keyboard(event, obj))
        self.entry_email.grid(column=1, row=2)
        self.input_pass = ""
        l_text = Label(frame2, text="Password", justify="left")
        l_text.grid(column=0, row=3)
        bullet = '\u2022'
        self.entry_password = Entry(frame2, width=20, textvariable=self.input_pass, show=bullet)
        self.entry_password.bind("<Button-1>", lambda event, obj=self.entry_password: self.cbk_keyboard(event, obj))
        self.entry_password.grid(column=1, row=3)
        l_text = Label(frame2, text="Gender", justify="left")
        l_text.grid(column=0, row=4)
        self.radio_variety = tkinter.StringVar()
        self.radio1 = tkinter.Radiobutton(frame2, text="Male", value="M", variable=self.radio_variety,
                                          command=lambda: self.cbk_check("male"))
        self.radio1.grid(column=1, row=4)
        self.radio2 = tkinter.Radiobutton(frame2, text="Female", value="F", variable=self.radio_variety,
                                          command=lambda: self.cbk_check("female"))
        self.radio2.grid(column=2, row=4)
        if self.radio_variety == 'M':
            self.radio2.deselect()
        else:
            self.radio1.deselect()
        input_birth = ""
        l_text = Label(frame2, text="Birth Day", justify="left")
        l_text.grid(column=0, row=5)
        self.entry_birth = Entry(frame2, width=20, textvariable=input_birth)
        self.entry_birth.bind("<Button-1>", lambda event, obj=self.entry_birth: self.cbk_keyboard(event, obj))
        self.entry_birth.grid(column=1, row=5)
        input_phone = ""
        l_text = Label(frame2, text="Phone", justify="left")
        l_text.grid(column=0, row=6)
        self.entry_phone = Entry(frame2, width=20, textvariable=input_phone)
        self.entry_phone.bind("<Button-1>", lambda event, obj=self.entry_phone: self.cbk_keyboard(event, obj))
        self.entry_phone.grid(column=1, row=6)
        if int(self.curr_user_id[0]) == 0:
            btn_signup = Button(frame2, text="Sign Up", command=lambda: self.cbk_signup(),
                                bg=self.curr_color, bd=1)
            btn_signup.grid(column=0, row=7)
        else:
            self.entry_fir_name.insert(0, self.mc.cur_user_info["first_name"])
            self.entry_last_name.insert(0, self.mc.cur_user_info["last_name"])
            self.entry_phone.insert(0, self.mc.cur_user_info["phone_number"])
            self.entry_email.insert(0, self.mc.cur_user_info["email"])
            self.entry_birth.insert(0, self.mc.cur_user_info["birth_date"].strftime("%Y%m%d"))
            self.entry_password.insert(0, self.mc.cur_user_info['password'])
            if self.mc.cur_user_info['gender'] == "male":
                self.radio1.select()
            else:
                self.radio2.select()
            btn_modify = Button(frame2, text="Modify", command=lambda: self.cbk_modi_user(),
                                bg=self.curr_color, bd=1)
            btn_modify.grid(column=0, row=7)

        frame3 = tk.Frame(self.win_setting)
        notebook.add(frame3, text="Photo")
        l_text = Label(frame3, text="Photo Interval Time(Sec): ", justify="left")
        l_text.place(rely=0.01, relx=0.05)

        self.var_interval = tk.StringVar(frame3)
        self.var_interval.set(self.mc.pt.interval)
        self.sb_interval = tk.Spinbox(frame3, from_=1, to=10, textvariable=self.var_interval, state="readonly")
        self.sb_interval.place(rely=0.1, relx=0.05)

        btn_acc = Button(frame3, width=5, height=1, text="Accept", command=lambda: self.cbk_change_interval(),
                         bg=COLOR[3])
        btn_acc.place(rely=0.1, relx=0.7)

        frame1 = tk.Frame(self.win_setting)
        notebook.add(frame1, text="Theme")

        self.btn_red = Button(frame1, width=8, height=3, text="RED", command=lambda: self.cbk_change_color(COLOR[0]),
                              bg=COLOR[0], borderwidth=0)
        self.btn_green = Button(frame1, width=8, height=3, text="Green",
                                command=lambda: self.cbk_change_color(COLOR[1]),
                                activebackground="#123345", bg=COLOR[1], borderwidth=0)
        self.btn_blue = Button(frame1, width=8, height=3, text="Blue", command=lambda: self.cbk_change_color(COLOR[2]),
                               activebackground="#123345", bg=COLOR[2], borderwidth=0)
        self.btn_gray = Button(frame1, width=8, height=3, text="Gray", command=lambda: self.cbk_change_color(COLOR[3]),
                               activebackground="#123345", bg=COLOR[3], borderwidth=0)
        self.btn_red.place(rely=0.01, relx=0.05)
        self.btn_green.place(rely=0.21, relx=0.05)
        self.btn_blue.place(rely=0.41, relx=0.05)
        self.btn_gray.place(rely=0.61, relx=0.05)

    def cbk_keyboard(self, event, obj):
        if self.window_keyboard == None:
            self.window_keyboard = tk.Toplevel()
            self.window_keyboard.overrideredirect(1)
            self.is_opened_oth_pg = True
            self.open_win_keyboard = True
            self.vk = Keyboard(self.window_keyboard)
            self.vk.add_entry(obj)
            self.vk.pack()
            self.window_keyboard.bind("<Destroy>", self._destroy_keyboard)
        else:
            self.vk.add_entry(obj)

    def cbk_change_interval(self):
        self.mc.pt.interval = int(self.sb_interval.get())

    def cbk_capture(self, id=0):
        self.mc.save_img(self.frame, user_id=id, mode="manual")

    def cbk_check(self, g):
        self.input_gender = g
        if g == "male":
            self.radio2.deselect()
        else:
            self.radio1.deselect()

    def cbk_modi_user(self):
        if self.radio_variety == "M":
            self.input_gender = "male"
            self.radio2.deselect()
            self.radio1.select()
        else:
            self.input_gender = "female"
            self.radio1.deselect()
            self.radio2.select()

        self.dict_user = {"password": self.entry_password.get(), "gender": self.input_gender,
                          "birth_date": self.entry_birth.get(),
                          "first_name": self.entry_fir_name.get(), "last_name": self.entry_last_name.get(),
                          "phone_number": self.entry_phone.get(), "email": self.entry_email.get(), "ap_id": '0'}
        self.mc.update_user_info(self.dict_user)

    def gather_data(self):
        if self.btn_state['Aging']:
            result_factor = []
            factors = self.mc.predict_age_factor(self.curr_face)
            if factors is not None:
                pred_age = np.mean(list(factors.values()))
                for f in factors:
                    result_factor.append("    " + f + ": " + str(factors[f]))
                self.set_txt(["[Aging Appearance]", "    Detected Age:" + str(round(pred_age)), "",
                              "[Aging Diagnosis]"] + result_factor)
            else:
                # self.set_txt(["[Aging Appearance]", "    Can't be measured.", "",
                #               "[Aging Diagnosis]", "    Can't be measured"])
                pass
        elif self.btn_state["Emotion"]:
            pred_emotion = self.mc.predict_emotion(self.curr_face)
            result_factor = []
            factors = self.mc.predict_emotion_factor()

            if factors is not None:
                for f in factors:
                    result_factor.append("    " + f + ": " + str(factors[f]))

                self.set_txt(["[Emotion Appearance]", "    Detected Emotion: " + str(pred_emotion)
                                 , "", "[Emotion Diagnosis]"] + result_factor)
            else:
                # self.set_txt(["[Emotion Appearance]", "    Can't be measured", "",
                #               "[Emotion Diagnosis]", "    Can't be measured"])
                pass
        elif self.btn_state["Recommend"]:
            recmm = ["[Age Advisory]"] + self.mc.get_recm()
            self.set_txt(recmm)

    def cbk_txt(self, idx):
        if not self.btn_state[idx]:
            self.btn_anal.configure(bg="#cccccc")
            self.btn_recm.configure(bg="#cccccc")
            self.btn_evol.configure(bg="#cccccc")
            if idx == "Aging":
                self.txt.delete(1.0, tk.END)
                self.txt.insert("current", "Computing age... \n")
                self.btn_anal.configure(bg="#675435")
                self.btn_state["Emotion"] = False
                self.btn_state["Recommend"] = False
            elif idx == "Emotion":
                self.txt.delete(1.0, tk.END)
                self.txt.insert("current", "Computing emotion... \n")
                self.btn_recm.configure(bg="#675435")
                self.btn_state["Aging"] = False
                self.btn_state["Recommend"] = False
            elif idx == "Recommend":
                self.txt.delete(1.0, tk.END)
                self.txt.insert("current", "Computing recommendation... \n")
                self.btn_evol.configure(bg="#675435")
                self.btn_state["Emotion"] = False
                self.btn_state["Aging"] = False
        else:
            if idx == "Aging":
                self.set_txt(txt=None)
                self.btn_anal.configure(bg="#cccccc")
            elif idx == "Emotion":
                self.set_txt(txt=None)
                self.btn_recm.configure(bg="#cccccc")
            elif idx == "Recommend":
                self.set_txt(txt=None)
                self.btn_evol.configure(bg="#cccccc")
        self.btn_state[idx] = not self.btn_state[idx]

    def cbk_signup(self):
        if self.radio_variety == "M":
            self.input_gender = "male"
            self.radio2.deselect()
            self.radio1.select()
        else:
            self.input_gender = "female"
            self.radio1.deselect()
            self.radio2.select()
        self.dict_user = {"password": self.entry_password.get(), "gender": self.input_gender,
                          "birth_date": self.entry_birth.get(),
                          "first_name": self.entry_fir_name.get(), "last_name": self.entry_last_name.get(),
                          "phone_number": self.entry_phone.get(), "email": self.entry_email.get(), "ap_id": '0'}
        # DB Connection Part
        result = self.db_user.register_user(self.dict_user)
        if result:
            curr_id = self.db_user.retrieve_user({"phone_number": self.entry_phone.get()})
            self.mc.set_session(True, curr_id[0]['id'])
            self.is_session = True
            self.mc.copy_files('.\\dataset\\train\\0', '.\\dataset\\train\\' + str(curr_id[0]['id']))
            self.fr.train_model()
            self.win_setting.destroy()

    def cbk_change_color(self, color):
        self.btn_plus.configure(bg=color)
        self.btn_minus.configure(bg=color)
        self.btn_capture.configure(bg=color)
        self.btn_setting.configure(bg=color)
        self.btn_recm.configure(bg=color)
        self.btn_anal.configure(bg=color)
        self.btn_evol.configure(bg=color)

    def cbk_click(self, event):
        if not self.is_opened_oth_pg:
            if not self.is_session or (self.is_session and self.fd.nrof_faces > 1):
                self.selected_x = event.x
                self.selected_y = event.y
        else:
            if self.open_win_progress:
                self.win_progress.destroy()
            if self.open_win_keyboard:
                self.window_keyboard.destroy()
            if self.open_win_setting:
                self.win_setting.destroy()

    def cbk_zoom_reset(self, event):
        self.curr_scale = 0

    def cbk_plus(self):
        if self.curr_scale < len(SCALE) - 1:
            self.curr_scale += 1

    def cbk_minus(self):
        if self.curr_scale > 0:
            self.curr_scale -= 1

    def cbk_info_view(self, event):
        if self.btn_state["Aging"]:
            self.draw_progress_view("Aging")
        elif self.btn_state["Emotion"]:
            self.draw_progress_view("Emotion")

    def _reset_session(self):
        if self.mc.get_session():
            self.mc.set_session(False, int(self.curr_user_id[0]))
            self.is_session = False
            self.mc.remove_img()
        self._remove_info()
        self.curr_user_id = ("", "")
        self.last_greeting_time = 0
        self.disappeared_time = 0
        self.is_said = False
        self.selected_y, self.selected_x = None, None
        self.remove_btn_fnc()
        self.remove_info_view()

    def _destroy_progress(self, event):
        self.is_opened_oth_pg = False
        self.open_win_progress = False

    def _destroy_setting(self, event):
        self.is_opened_oth_pg = False
        self.open_win_setting = False

    def _destroy_keyboard(self, event):
        self.window_keyboard = None
        self.is_opened_oth_pg = False
        self.open_win_keyboard = False

    def _remove_info(self):
        self.txt.delete(1.0, tk.END)

    def _check_skip(self, dtt_usr_id):
        if self.is_session:
            if (
                    self.fd.nrof_faces == self.last_num and self.curr_user_id != dtt_usr_id) or self.fd.nrof_faces != self.last_num:
                if self.skip_count < 5:
                    self.skip_count += 0
                else:
                    self.is_skipped = False
                    self._reset_session()

            elif self.fd.nrof_faces == self.last_num and self.curr_user_id == dtt_usr_id:
                if self.skip_count > 0:
                    self.skip_count -= 1
        else:
            self.is_skipped = False


if __name__ == '__main__':
    ui = UserInterface()
    ui.run()
