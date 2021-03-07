from threading import Thread

from pyowm import OWM
import pdairp
from datetime import datetime
import time
from ActiveAgingAdvisorySystem.progressanalyzer import EmotionProgressAnalyzer, AgingProgressAnalyzer


class AgingSummary:
    weather_api_key = '74680e679a7e0478da2d8548a01ad426'

    def __init__(self):
        self.city = 'Seoul,kr'
        self.lat = 37.0
        self.lon = 126.0
        self.w = []
        self.today_weather = {}
        self.is_load = True
        self.owm = OWM(self.weather_api_key)
        self.obs = self.owm.weather_at_place(self.city)
        self.uvc = self.owm.uvindex_around_coords(self.lat, self.lon)
        self.load_today_weather()

        self.epa = EmotionProgressAnalyzer()
        self.apa = AgingProgressAnalyzer()

    def set_location(self, loc, lat, lon):
        """
        To set target location
        :param loc: string, city name
        :param lat: float, latitude
        :param lon: float, longitude
        :return:
        """
        self.city = loc
        self.lat = lat
        self.lon = lon

    def load_today_weather(self):
            self.w = self.obs.get_weather()
            self.weather = self.w.get_status()
            self.temper = self.w.get_temperature(unit="celsius")['temp']
            self.humid = self.w.get_humidity()
            self.uvi = self.uvc.get_value()

    def get_today_weather(self):
        self.today_weather['Weather'] = self.get_weather()
        self.today_weather['Temperature'] = self.get_temperature()
        self.today_weather['Humidity'] = self.get_humidity()
        self.today_weather['UV Index'] = self.get_uvi()

        return self.today_weather

    def get_uvi(self):
        return self.uvi

    def get_humidity(self):
        return self.humid

    def get_temperature(self):
        return self.temper

    def get_weather(self):
        return self.weather

    def set_curr_id(self, id):
        self.curr_id = id

    def aging_summary(self, d):
        """
        To return aging summary
        :return:
        """
        result = self.apa.get_seasonal_variation_of_age_pre(self.curr_id, d)
        f, l = result['formal average age'], result['latter average age']
        return f, l

    def emotion_summary(self, d):
        """
        To return emotion summary
        :return:
        """
        result = self.epa.get_top_emotions(self.curr_id, d)
        f, l = result["start_top_emotion"], result['end_top_emotion']
        # f, l = "Angry", "Happy"
        return f, l

    def get_summary_tips(self):
        return ""


if __name__ == '__main__':
    aging_smm = AgingSummary()
    aging_smm.load_today_weather()
    print(aging_smm.weather)