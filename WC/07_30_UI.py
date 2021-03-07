from dateutil.relativedelta import relativedelta
import time
import tkinter as tk
import tkinter.font
import WC.progressanalyzer as pa

from datetime import datetime
from datetime import timedelta

import tkinter.ttk

from tkinter import Label, PhotoImage, Button, Entry
import cv2
from PIL import Image, ImageTk

# sys.path.append(".\\JC")
COLOR = ["#efb5ae", "#c8e7a7", "#7fd7f7", "#CCCCCC"]
SCALE = [1.0, 0.8, 0.6, 0.4, 0.2]


class UserInterface:

    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.btn_state = {"Aging": True, "Emotion": False, "Recommend": False}


        self.width, self.height = int(self.cap.get(3)*1.4), int(self.cap.get(4)*1.4)
        # self.width, self.height = int(self.cap.get(3)), int(self.cap.get(4))
        self.curr_color = COLOR[3]
        self.root = tk.Tk()
        # self.root.overrideredirect(1)
        # self.root.attributes("-fullscreen", True)     # Full Screen
        # self.root.attributes("")
        # self.set_widow_size(self.width, self.height)
        self.root.bind('<Escape>', lambda e: self.root.quit())
        # self.root.bind('<Button-1>', self.cbk_click)
        self.lmain = tk.Label(self.root)
        self.lmain.pack()
        self.start_time = time.time()

        # self.photo_taker = self.mc.pt
        self.is_session = False
        self.user_id = 0
        self.frame = None
        # self.db_user = self.mc.usermng

        self.selected_x, self.selected_y = None, None
        self.curr_user_id = None
        self.curr_face = None

        self.greeting_duration = 5
        self.curr_scale = 0
        self.last_num = 0

        self.last_greeting_time = 0
        self.disappeared_time = 0
        self.is_said = False
        self.do_draw_age = False

        self.duration_progression = 5

    def cbk_info_view(self, event=None):

        if self.btn_state["Aging"]:
            graph_type = 'Aging'
            self.win_progress = tk.Toplevel()
            self.win_progress.geometry("1100x650+200+200")
            # self.win_progress.resizable(False, False)

            # 엔트리 배치
            self.ent_info_view()
            # 체크박스 배치
            self.chk_info_view(graph_type)
            # 레이블 배치
            self.lbl_info_view()
            # 버튼 배치
            self.btn_info_view(graph_type)

        elif self.btn_state["Emotion"]:
            graph_type = 'Emotion'
            self.win_progress = tk.Toplevel()
            self.win_progress.geometry("1100x650+200+200")
            # self.win_progress.resizable(False, False)

            # 엔트리 배치
            self.ent_info_view()
            # 체크박스 배치
            self.chk_info_view(graph_type)
            # 레이블 배치
            self.lbl_info_view()
            # 버튼 배치
            self.btn_info_view(graph_type)

        self.win_progress.mainloop()

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
        combo_interval = tk.ttk.Combobox(self.win_progress, width=20, textvariable=self.combo_str)
        combo_interval['values'] = ('Daily', 'Weekly', 'Monthly', 'Quarterly', 'Yearly')
        combo_interval.place(relx=0.09, rely=0.18, width=80, height=30)

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
            hist, pie = pa.AgingProgressAnalyzer().get_trend_of_age(2, 'WOO CHAN',
                                                                    dict_duration=dict_duration,
                                                                    interval=interval,
                                                                    dict_fact=dict_fact)

        elif graph_type == 'Emotion':
            # hist, pie = self.mc.get_emotion_charts()
            hist, pie = pa.EmotionProgressAnalyzer().get_trend_of_emotions(2, 'WOO CHAN',
                                                                    dict_duration=dict_duration,
                                                                    interval=interval,
                                                                    dict_fact=dict_fact)
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


UserInterface().cbk_info_view()
