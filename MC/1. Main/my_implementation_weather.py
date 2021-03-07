import requests


class Weather:
    s_city = "Seoul,KR"
    appid = "7b88b3ab4ea349e89cca25ca0576fa0f"

    def on(self):
        self.find_city_id()
        self.current_weather()
        self.forecast_5days()

    def find_city_id(self):
        try:
            res = requests.get("http://api.openweathermap.org/data/2.5/find",
                               params={'q': self.s_city, 'type': 'like', 'units': 'metric', 'APPID': self.appid})
            data = res.json()
            cities = ["{} ({})".format(d['name'], d['sys']['country'])
                      for d in data['list']]
            print('Location information:')
            print("city:", cities)
            self.city_id = data['list'][0]['id']
            print('city_id =', str(self.city_id) + '\n')

            return self.city_id
        except Exception as e:
            print("Exception (find):", e)
            pass

    def current_weather(self):
        try:
            res = requests.get("http://api.openweathermap.org/data/2.5/weather",
                               params={'id': self.city_id, 'units': 'metric', 'lang': 'en', 'APPID': self.appid})
            data = res.json()
            print('Current weather:')
            print("Temp:", str(data['main']['temp']) + '\u00b0C')
            print("Temp min:", str(data['main']['temp_min']) + '\u00b0C')
            print("Temp max:", str(data['main']['temp_max']) + '\u00b0C')
            print("Conditions:", data['weather'][0]['description'].title())
            print("Humidity:", str(data['main']['humidity']) + ' %')
            print("Pressure :", str(data['main']['pressure']) + ' hPa')
            print("Wind speed:", str(data['wind']['speed']) + ' m/s')
        except Exception as e:
            print("Exception (weather):", e)
            pass

    def forecast_5days(self):
        try:
            res = requests.get("http://api.openweathermap.org/data/2.5/forecast",
                               params={'id': self.city_id, 'units': 'metric', 'lang': 'en', 'APPID': self.appid})
            data = res.json()
            print('\nForecast for 5 days:')
            for i in data['list']:
                print(i['dt_txt'], '{0:+3.0f}'.format(i['main']['temp']), i['weather'][0]['description'].title())
        except Exception as e:
            print("Exception (forecast):", e)
            pass


weather = Weather()
weather.on()


# Location information:
# city: ['Seoul (KR)']
# city_id = 1835848
#
# Current weather:
# Temp: 23.54°C
# Temp min: 22°C
# Temp max: 25°C
# Conditions: Haze
# Humidity: 78 %
# Pressure : 1009 hPa
# Wind speed: 0.5 m/s
#
# Forecast for 5 days:
# 2019-07-12 03:00:00 +23 Few Clouds
# 2019-07-12 06:00:00 +23 Clear Sky
# 2019-07-12 09:00:00 +23 Clear Sky
# 2019-07-12 12:00:00 +22 Light Rain
# 2019-07-12 15:00:00 +21 Light Rain
# 2019-07-12 18:00:00 +21 Light Rain
# 2019-07-12 21:00:00 +21 Light Rain
# 2019-07-13 00:00:00 +21 Overcast Clouds
# 2019-07-13 03:00:00 +21 Overcast Clouds
# 2019-07-13 06:00:00 +22 Broken Clouds
# 2019-07-13 09:00:00 +22 Few Clouds
# 2019-07-13 12:00:00 +22 Clear Sky
# 2019-07-13 15:00:00 +22 Clear Sky
# 2019-07-13 18:00:00 +21 Clear Sky
# 2019-07-13 21:00:00 +21 Clear Sky
# 2019-07-14 00:00:00 +21 Clear Sky
# 2019-07-14 03:00:00 +22 Light Rain
# 2019-07-14 06:00:00 +22 Clear Sky
# 2019-07-14 09:00:00 +22 Clear Sky
# 2019-07-14 12:00:00 +22 Clear Sky
# 2019-07-14 15:00:00 +22 Light Rain
# 2019-07-14 18:00:00 +22 Moderate Rain
# 2019-07-14 21:00:00 +21 Light Rain
# 2019-07-15 00:00:00 +22 Light Rain
# 2019-07-15 03:00:00 +22 Few Clouds
# 2019-07-15 06:00:00 +22 Few Clouds
# 2019-07-15 09:00:00 +22 Clear Sky
# 2019-07-15 12:00:00 +22 Clear Sky
# 2019-07-15 15:00:00 +22 Clear Sky
# 2019-07-15 18:00:00 +21 Clear Sky
# 2019-07-15 21:00:00 +21 Clear Sky
# 2019-07-16 00:00:00 +22 Clear Sky
# 2019-07-16 03:00:00 +22 Clear Sky
# 2019-07-16 06:00:00 +23 Clear Sky
# 2019-07-16 09:00:00 +24 Clear Sky
# 2019-07-16 12:00:00 +23 Clear Sky
# 2019-07-16 15:00:00 +23 Broken Clouds
# 2019-07-16 18:00:00 +22 Light Rain
# 2019-07-16 21:00:00 +22 Overcast Clouds
# 2019-07-17 00:00:00 +23 Overcast Clouds