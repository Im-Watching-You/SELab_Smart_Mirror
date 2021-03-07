from __future__ import print_function
import pickle
import os.path
import pdairp
from pyowm import OWM
from datetime import datetime
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from apiclient import errors
import threading


class Personal:


    SCOPES_calendar = ['https://www.googleapis.com/auth/calendar.readonly']
    SCOPES_gmail = ['https://www.googleapis.com/auth/gmail.readonly']
    msgs_id = []
    msgs_id_threadId = []
    list_messages_subject = []

    # act_event = True

    # def on(self, id):
    #     self.act_event = True
    #     self.get_unread_messages(id)
    #     self.get_events(id)
    #     self.get_gmail_labels(id)
    #
    # def off(self):
    #     self.act_event = False

    def get_events(self, id):
        service = self.get_credentials('calendar', 'v3', self.SCOPES_calendar,
                                       'credentials_calendar.json', id, 'calendar')
        now = datetime.utcnow().isoformat() + 'Z'  # 'Z' indicates UTC time
        events_result = service.events().list(calendarId='primary', timeMin=now,
                                              maxResults=2, singleEvents=True,
                                              orderBy='startTime').execute()
        events = events_result.get('items', [])
        # print(events)
        # Show events with dateTime
        if not events:
            print('\nNo upcoming events found.\n')
        else:
            print('\nEvents from calendar:')
            for event in events:
                start = event['start'].get('dateTime')
                print('Event: ' + str(event['summary']) + "\nData: " + str(start) + '\n')
        return events

    def get_gmail_labels(self, id):    # Call the Gmail API to get gmail_labels
        service = self.get_credentials('gmail', 'v1', self.SCOPES_gmail, 'credentials_gmail.json', id, 'gmail')
        results = service.users().labels().list(userId='me').execute()
        labels = results.get('labels', [])

        if not labels:
            print('No labels found.')
        else:
            print('Labels:')
            for label in labels:
                print(label['name'])
        return labels

    def get_unread_messages(self, id):
        try:
            service = self.get_credentials('gmail', 'v1', self.SCOPES_gmail, 'credentials_gmail.json', id, 'gmail')
            # To get unread messages_ID
            response = service.users().messages().list(userId='me', q='is:unread').execute()
            if 'messages' in response:
                self.msgs_id_threadId.extend(response['messages'])

            for i in self.msgs_id_threadId:
                self.msgs_id.append(i['id'])

            # To get unread messages
            list_messages_subject = []
            if not self.msgs_id:
                print('\nNo new messages!\n')
            else:
                # snippet = body_text; message-payload-headers[16][value] = subject
                # list_messages = []
                # print('Message snippet: %s' % message['snippet'])
                # list_messages.append(message['snippet'])
                for msg_id in self.msgs_id:
                    message = service.users().messages().get(userId='me', id=msg_id, format='metadata').execute()
                    subject = message.get('payload').get('headers')[16]['value']
                    print('Message subject: %s' % subject)
                    self.list_messages_subject.append(subject)
            return self.list_messages_subject
        except errors.HttpError as error:
            print('An error occurred: %s' % error)

    def get_credentials(self, API, ver, SCOPES, jsonFile, id, FilePickleName):    # Get credentials from API
        creds = None
        # To read credentials from file
        token_file_name = ".\\user\\" + "token_" + str(id) + "_" + FilePickleName + ".pickle"
        if os.path.exists(token_file_name):
            with open(token_file_name, 'rb') as token:
                creds = pickle.load(token)

        # If there are no credentials, let the user log in and
        if not creds or not creds.valid:
            print("Please authorize your google account.")
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(jsonFile, SCOPES)  # To call authorization.
                creds = flow.run_local_server()

            with open(token_file_name, 'wb') as token:      # Save the credentials for the next run
                pickle.dump(creds, token)
        return build(API, ver, credentials=creds)


class Public:
    time_info = None
    weather_info = 'Loading...'
    min_max_info = None
    w = None
    rain_vol = None
    snow_vol = None

    s = ""
    c = ""
    temp = ""
    min_temp = ""
    max_temp = ""
    alert_msg = []

    pm10_state = ""
    pm25_state = ""
    pm10_density = None
    pm25_density = None
    pm10_info = None
    pm25_info = None

    weather_state = ""
    dust_state = ""

    weather_api_key = '74680e679a7e0478da2d8548a01ad426'
    dust_api_key = '3BOiwUkdMKq%2BgIDykKukmY%2BHqh8DHVzB3anLiWbZmKkP%2Bg6TDGTck7mQIQTQuQ0x7b9JVLwOMcsliCKUU8RGAg%3D%3D'

    owm = OWM(weather_api_key)
    city = 'Seoul,kr'
    obs = owm.weather_at_place(city)

    duse_api = pdairp.PollutionData(dust_api_key)
    dust = duse_api.station('동작구', 'DAILY', page_no='1', num_of_rows='10', ver='1.2')['0']

    def on(self):
        print(self.get_time())
        print(self.get_weather_state())
        print(self.get_dust_state())

    def off(self):
        self.time_info = None
        self.weather_info = 'Loading...'
        self.pm10_density = None
        self.pm25_density = None
        self.pm10_info = None
        self.pm25_info = None
        self.w = None

    def get_time(self):
        self.time_info = datetime.today().strftime('%Y-%m-%d %H:%M:%S\n')
        return self.time_info

    def get_weather_state(self):
        if self.w is None:
            self.alert_msg = []
            self.w = self.obs.get_weather()
            self.s = self.w.get_detailed_status()
            self.temp = self.w.get_temperature(unit='celsius')
            self.c = self.temp['temp']
            self.min_temp = self.temp['temp_min']
            self.max_temp = self.temp['temp_max']
            self.weather_info = '{}: {}\u00b0C | {}'.format("Seoul,kr", self.c, self.s.title())
            self.min_max_info = 'Max: {}\u00b0C | Min: {}\u00b0C'.format(self.max_temp, self.min_temp)
            self.weather_state = '{}\n{}\n'.format(self.weather_info, self.min_max_info)
            return self.weather_state

    def get_dust_state(self):
        if self.pm10_density is None and self.pm25_density is None:
            self.pm10_density = int(self.dust['pm10Value'])
            self.pm25_density = int(self.dust['pm25Value'])
            self.define_dust_level(self.pm10_density, self.pm25_density)
            self.pm10_info = 'Fine Dust: {} | {}'.format(self.pm10_density, self.pm10_state)
            self.pm25_info = 'Ultra-fine Dust: {} | {}'.format(self.pm25_density, self.pm25_state)
            self.dust_state = '{}\n{}\n'.format(self.pm10_info, self.pm25_info)
            return self.dust_state

    def define_dust_level(self, pm10_density, pm25_density):
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


public = Public()
public.on()
