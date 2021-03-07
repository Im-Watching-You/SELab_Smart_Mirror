import threading
import time

import logging

from pyowm import OWM
from db_connector import User, UserProfile, UserSession, UserSessionList
import mttkinter.mtTkinter as tk
#import tkinter as tk
from tkinter import messagebox
from virtual_keyboard import Keyboard
from flags import Flags, ButtonFlag
import os
import pdairp

from datetime import datetime
import pickle
import os.path

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

from assessment_legacy import EmotionDetector, AgeDetector
from google_assistant import google_ai_starter
from evolution_legacy import EvolutionAction
from smart_diary import Smart_Diary
from detector import Detector
import numpy as np
from collections import Counter


class Public:
    time_info = None
    weather_info = 'Loading...'
    min_max_info = None
    w = None
    rain_vol = None
    snow_vol = None

    s = ""
    temp = ""
    c = ""
    min_temp = ""
    max_temp = ""
    alert_msg = []

    pm10_state = ""
    pm10_density = None
    pm25_state = ""
    pm25_density = None
    pm10_info = None
    pm25_info = None

    weather_api_key = '74680e679a7e0478da2d8548a01ad426'
    dust_api_key = '3BOiwUkdMKq%2BgIDykKukmY%2BHqh8DHVzB3anLiWbZmKkP%2Bg6TDGTck7mQIQTQuQ0x7b9JVLwOMcsliCKUU8RGAg%3D%3D'

    owm = OWM(weather_api_key)
    city = 'Seoul,kr'
    obs = owm.weather_at_place(city)

    duse_api = pdairp.PollutionData(dust_api_key)
    dust = duse_api.station('동작구', 'DAILY', page_no='1', num_of_rows='10', ver='1.2')['0']

    def on(self):
        self.event_public()

    def off(self):
        self.time_info = None
        self.weather_info = 'Loading...'
        self.pm10_density = None
        self.pm25_density = None
        self.pm10_info = None
        self.pm25_info = None
        self.w = None

    def get_dust_state(self, pm10_density, pm25_density):
        if pm10_density <= 30:
            self.pm10_state = 'Good'
        elif 31 <= pm10_density <= 80:
            self.pm10_state = 'Normal'
        elif 81 <= pm10_density <= 100:
            self.pm10_state = 'Bad'
        else:
            self.pm10_state = 'Very Bad'
        if pm25_density <= 15:
            self.pm25_state = 'Good'
        elif 16 <= pm25_density <= 35:
            self.pm25_state = 'Normal'
        elif 36 <= pm25_density <= 75:
            self.pm25_state = 'Bad'
        else:
            self.pm25_state = 'Very Bad'

    def event_public(self):
        def th():
            if self.time_info is None:
                self.alert_msg = []
                self.w = self.obs.get_weather()
                self.s = self.w.get_detailed_status()
                self.temp = self.w.get_temperature(unit='celsius')
                self.c = self.temp['temp']
                self.min_temp = self.temp['temp_min']
                self.max_temp = self.temp['temp_max']
                self.weather_info = '{} : {} {}'.format("Seoul,kr", self.s, self.c)
                self.min_max_info = 'Max : {} | Min : {}'.format(self.max_temp, self.min_temp)

                if self.pm10_density is None and self.pm25_density is None:
                    self.pm10_density = int(self.dust['pm10Value'])
                    self.pm25_density = int(self.dust['pm25Value'])
                    self.get_dust_state(self.pm10_density, self.pm25_density)
                    self.pm10_info = 'Fine Dust : {} | {}'.format(self.pm10_density, self.pm10_state)
                    self.pm25_info = 'Ultra-fine Dust : {} | {}'.format(self.pm25_density, self.pm25_state)
            self.time_info = datetime.today().strftime('%Y-%m-%d %H:%M:%S')

        threading.Thread(target=th).start()


class Personal:
    act_event = True
    info = ""
    is_auth_called = False

    def on(self, id):
        self.act_event = True
        self.event_personal(id)

    def off(self):
        self.act_event = False

    def event_personal(self, id):
        SCOPES_calendar = ['https://www.googleapis.com/auth/calendar.readonly']
        SCOPES_gmail = ['https://www.googleapis.com/auth/gmail.readonly']
        def th(id):
            # try:
            creds = None
            # The file token.pickle stores the user's access and refresh tokens, and is
            # created automatically when the authorization flow completes for the first
            token_file_name = ".\\user\\"+"token_"+str(id)+"_calendar.pickle"
            if os.path.exists(token_file_name):

                with open(token_file_name, 'rb') as token:
                    creds = pickle.load(token)
            # If there are no (valid) credentials available, let the user log in.

            if not self.is_auth_called and (creds is None or not creds.valid):
                self.info = "Please authorize your google account."
                self.is_auth_called = True
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        'credentials_calendar.json', SCOPES_calendar)             # To call authorization.
                    creds = flow.run_local_server()
                # Save the credentials for the next run
                with open(token_file_name, 'wb') as token:
                    pickle.dump(creds, token)

            if creds is not None:
                service = build('calendar', 'v3', credentials=creds)

                # Call the Calendar API
                now = datetime.utcnow().isoformat() + 'Z'  # 'Z' indicates UTC time
                events_result = service.events().list(calendarId='primary', timeMin=now,
                                                      maxResults=2, singleEvents=True,
                                                      orderBy='startTime').execute()
                events = events_result.get('items', [])

                if not events:
                    self.info = 'No upcoming events found.'
                else:
                    for event in events:
                        start = event['start'].get('dateTime', event['start'].get('date'))
                        self.info = str(str(start)+" "+str(event['summary']))+"\n"

                creds = None
                # The file token.pickle stores the user's access and refresh tokens, and is
                # created automatically when the authorization flow completes for the first
                # time.
                token_file_name = ".\\user\\" + "token_" + str(id) + "_gmail.pickle"
                if os.path.exists(token_file_name):
                    with open(token_file_name, 'rb') as token:
                        creds = pickle.load(token)
                # If there are no (valid) credentials available, let the user log in.
                if not creds or not creds.valid:
                    if creds and creds.expired and creds.refresh_token:
                        creds.refresh(Request())
                    else:
                        flow = InstalledAppFlow.from_client_secrets_file(
                            'credentials_gmail.json', SCOPES_gmail)
                        creds = flow.run_local_server()
                    # Save the credentials for the next run
                    with open(token_file_name, 'wb') as token:
                        pickle.dump(creds, token)

                service = build('gmail', 'v1', credentials=creds)

                # Call the Gmail API
                results = service.users().labels().list(userId='me').execute()
                labels = results.get('labels', [])

                if not labels:
                    print('No labels found.')
                else:
                    print('Labels:')
                    for label in labels:
                        print(label['name'])

            time.sleep(1)
            # except:
            #     pass

        threading.Thread(target=th, args=(id,)).start()


class Assess:
    act_event = True
    emotion_info = None
    age_info = None
    user_session = UserSession()
    session = False

    def __init__(self):
        self.emotion = EmotionDetector()
        self.age = AgeDetector()

    def on(self, face_frame, user_id='0001'):
        if face_frame is not None:
            self.act_event = True
            self.event_assess(face_frame, user_id)
            self.session = True

    def off(self):
        self.act_event = False
        # if self.session:
            # self.user_session.close()

    def event_assess(self, face_frame, user_id):
        result_emotion = self.emotion.detect(face_frame)
        result_age = self.age.detect(face_frame)
        # self.user_session.insert_assessment(user_id, result_emotion, int(result_age))
        #print(">>> ", result_age, result_emotion)
        self.emotion_info = result_emotion
        self.age_info = str(result_age)


class Suggest:
    act_event = True

    def on(self):
        self.act_event = True
        self.event_suggest()

    def off(self):
        self.act_event = False

    def event_suggest(self):
        def th():
            while self.act_event:
                #print("Suggest")
                time.sleep(1)
        threading.Thread(target=th).start()


class Evolution:
    act_event = True
    evolution_info = None
    evolution_charts = []

    def __init__(self):
        self.us = UserSessionList()
        self.evol = EvolutionAction()

    def on(self):
        self.act_event = True
        self.usr_id=None
        self.list_data = None

    def off(self):
        self.act_event = False

    def set_usr_id(self, usr_id):
        self.usr_id = usr_id

    def event_evolution(self):
        list_data = self.us.load_session_list_db(self.usr_id)
        most = self.evol.compute_most_value_cand(list_data, 5)   # To get computing most value
        com = self.evol.compare_data_cand(list_data, 5)      # To get comparison data
        most.update(com)        # To combine both data
        self.evolution_info = most

        print(most, com)
        com = self.evol.compare_whole_data(list_data, 10)      # To get comparison data
        # self.evolution_charts = self.evol.draw_plot(com['com_oth_emotion'], com['com_tod_emotion'],
        #                                             com['com_oth_age'], com['com_tod_age'])
        self.evolution_charts = self.evol.draw_plot_sequence(com['emotions'], com['age'])


class Diary:
    act_event = True
    sd = Smart_Diary()

    first_record = True
    record = False

    def on(self):
        self.act_event = True

    def off(self):
        self.act_event = False
        self.first_record = True
        self.record = False

    def event_recording(self):
        if self.first_record:
            self.sd.record_start()
            self.first_record = False
            self.record = True

    def event_reading(self):
        self.sd.update_info_db('harim')
        return self.sd.get_diary_data()


class Setting:
    act_event = True

    def on(self):
        self.act_event = True
        self.event_setting()

    def off(self):
        self.act_event = False

    def event_setting(self):
        def th():
            while self.act_event:
                time.sleep(1)
        threading.Thread(target=th).start()


class Assistant:
    act_event = False
    first_act = False

    def on(self):
        self.act_event = True
        if self.first_act:
            pass
        else:
            self.event_assistant()
            self.first_act = True

    def off(self):
        self.act_event = False
        self.first_act = False

    def event_assistant(self):
        threading.Thread(target=google_ai_starter).start()


class Join:
    def __init__(self):
        self.act_event = False
        self.first_act = False
        self.title = "Registration"
        self.vk = None
        self.keyboard = None
        self.event_entry = None

    def on(self):
        self.act_event = True
        if self.first_act:
            pass
        else:
            self.ask_join()
            self.first_act = True

    def off(self):
        self.__init__()
        self.first_act = False
        ButtonFlag.register_flag = False

    def ask_join(self):
        def a():
            root = tk.Tk()
            root.withdraw()
            ask_btn = messagebox.askquestion(self.title, message='Welcome to Smart Mirror\nWould you like to register?')
            if ask_btn == 'yes':
                self.event_join()
                root.destroy()
            else:
                root.destroy()
                Flags.Joining = False
        threading.Thread(target=a).start()

    def event_join(self):
        def get_info(event):
            first_name = first_name_entry.get()
            last_name = last_name_entry.get()
            email = email_entry.get()
            phone_num = phone_num_entry.get()
            print(first_name, last_name, email, phone_num)
            user_info = dict(first_name=first_name, last_name=last_name,
                                  email=email, phone_number=phone_num)
            user = User()
            user.create(user_info)
            curr_id = user.read_id()
            os.mkdir(".\\image\\"+str(curr_id)+"\\")
            d = Detector()
            d.train_model(curr_id)

            print('hello, {} {}'.format(user.info['first_name'], user.info['last_name']))
            if self.keyboard is not None:
                self.keyboard.destroy()
                self.keyboard = None
            if self.vk is not None:
                self.vk.destroy()
                self.vk = None
            window.destroy()
            Flags.Joining = False

        def close(event):
            if self.keyboard is not None:
                self.keyboard.destroy()
                self.keyboard = None
            if self.vk is not None:
                self.vk.destroy()
                self.vk = None
            window.destroy()
            Flags.Joining = False

        def input_key(event):
            if self.event_entry is event.widget:
                pass
            else:
                if self.keyboard is not None:
                    self.keyboard.add_entry(event.widget)
                else:
                    self.use_vk = True
                    self.vk = tk.Tk()
                    self.vk.overrideredirect(1)
                    self.vk.geometry("450x250+500+400")
                    self.keyboard = Keyboard(self.vk)
                    self.keyboard.add_entry(event.widget)
                    self.keyboard.pack()
                    self.vk.mainloop()
                self.event_entry = event.widget

        msg = "Registration of Smart Mirror System"
        window = tk.Tk(className='Registration')
        window.geometry("400x180+500+200")
        msg_label = tk.Label(window, text=msg)
        first_name_label = tk.Label(window, text='First Name')
        first_name_entry = tk.Entry(window, bd=5)
        first_name_entry.bind('<FocusIn>', input_key)
        last_name_label = tk.Label(window, text='Last Name')
        last_name_entry = tk.Entry(window, bd=5)
        last_name_entry.bind('<FocusIn>', input_key)
        email_label = tk.Label(window, text='email')
        email_entry = tk.Entry(window, bd=5)
        email_entry.bind('<FocusIn>', input_key)
        phone_num_label = tk.Label(window, text='Phone Number')
        phone_num_entry = tk.Entry(window, bd=5)
        phone_num_entry.bind('<FocusIn>', input_key)
        yes_btn = tk.Button(window, text='OK', overrelief='groove', bg='green2')
        yes_btn.bind('<Button-1>', get_info)
        no_btn = tk.Button(window, text='Cancel', overrelief='groove', bg='red')
        no_btn.bind('<Button-1>', close)

        msg_label.grid(row=0, column=1)
        first_name_label.grid(row=1, column=0)
        first_name_entry.grid(row=1, column=1)
        last_name_label.grid(row=2, column=0)
        last_name_entry.grid(row=2, column=1)
        email_label.grid(row=3, column=0)
        email_entry.grid(row=3, column=1)
        phone_num_label.grid(row=4, column=0)
        phone_num_entry.grid(row=4, column=1)
        yes_btn.grid(row=5, column=0)
        no_btn.grid(row=5, column=1)
        window.mainloop()




if __name__ == '__main__':
    personal = Personal()
    personal.on()