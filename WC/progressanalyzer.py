import WC.joinedtable as jt

import datetime
from datetime import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta

import matplotlib.pyplot as plt

import random
from math import ceil

import cv2

import operator
import numpy as np
import pandas as pd

class ProgressAnalyzer:

    def _get_seasonal_variation_of_factors(self, user_id, assess_type, table_type, rank=3):
        """
        To get each factor's variation
        :param user_id: Integer
        :param assess_type: String, one of ['aging_diagnosis', 'emotion_diagnosis']
        :param table_type: String, one of ['spaaf', 'spaef']
        :param rank: Integer, {1<=rank<=10}
        :return: result : List of Dictionaries or Empty List
        """
        if table_type not in ['spaaf', 'spaef']:
            print('Invalid table type.')
            return [{}]

        # There are 10 factors for detecting emotions
        if rank < 1 or rank > 10:
            print("invalid rank value.")
            return [{}]

        start_date_rows, end_date_rows = self._retrieve_user_variation(user_id, assess_type, table_type)

        # print(start_date_rows, end_date_rows)

        # When retrieving method returns nothing.
        if start_date_rows == [] and end_date_rows == []:
            return [{}]

        # To calculate sum of each date's factor rate
        if table_type == 'spaaf':
            dict_start_sum, dict_end_sum = self._calculate_age_sum(start_date_rows, end_date_rows)

        elif table_type == 'spaef':
            dict_start_sum, dict_end_sum = self._calculate_emotion_sum(start_date_rows, end_date_rows)

        else:
            print('ERROR: WC.progress analyzer._get_seasonal_variation_of_factors')
            return [{}]

        # To calculate average rate of each factors
        for i in dict_start_sum:
            # print(type(dict_start_sum[i]))
            # print(dict_start_sum[i])
            dict_start_sum[i] /= len(start_date_rows)

        for i in dict_end_sum:
            # print(end_date_rows)
            # print(len(end_date_rows))
            dict_end_sum[i] = round(dict_end_sum[i] / len(end_date_rows), 2)

        # 'sorted' method returns tuple type
        start_sorted_factor = sorted(dict_start_sum.items(), key=operator.itemgetter(1), reverse=True)[0:rank]

        result = []

        # print(dict_start_sum)
        # print(dict_end_sum)

        # To make result list
        for i in range(0, rank):
            # print(start_sorted_factor)
            factor_name = start_sorted_factor[i][0]

            try:
                factor_rate = round(dict_end_sum[f'{factor_name}'] / dict_start_sum[f'{factor_name}'], 2)
                # print(factor_rate)

            except ZeroDivisionError:
                if dict_end_sum[f'{factor_name}'] !=0:
                    factor_rate = 2
                else:
                    factor_rate = 1

            except KeyError:
                print("This case happen when some detected factors which are in start_date factors but aren't in end_date_factors.")
                continue

            if factor_rate > 1.05:
                state = ['increasing', f'{round(abs(1 - factor_rate) * 100,0)}%']

            elif 1.05 >= factor_rate >= 0.95:
                state = [f'maintaining', '']

            else:
                state = [f'decreasing', f'{round(abs(1 - factor_rate) * 100,0)}%']

            result.append({f'{factor_name}': state})

        return result

    # origin code for factors
    def _retrieve_user_variation(self, user_id, assess_type, table_type):
        """
         To return two kind of rows
            (1) rows with start_date.
            (2) rows with end_date.

        :param user_id: Integer
        :param assess_type: String, one of ['aging_diagnosis', 'emotion_diagnosis']
        :param duration: String, one of ['week', 'month', 'quarter', 'year']
        :return: start_date_rows, end_date_rows
        """
        if self.start_date is None and self.end_date is None:
            print("\'sesonal_variation_of_()\' need to execute \'get_trend_of_()\' first.")
            return [], []

        dict_duration = {'start_date': self.start_date, 'end_date': self.end_date}

        # print(dict_duration)

        start_date_rows, end_date_rows = jt.SPATable().retrieve_user_variation(user_id, assess_type, table_type, dict_duration)

        if start_date_rows is not () and end_date_rows is not ():
            return start_date_rows, end_date_rows

        else:
            return [], []

    # test code
    def _retrieve_user_variation_tmp(self, user_id, assess_type):
        """
         To return two kind of rows
            (1) rows with start_date.
            (2) rows with end_date.

        :param user_id: Integer
        :param assess_type: String, one of ['aging_diagnosis', 'emotion_diagnosis']
        :param duration: String, one of ['week', 'month', 'quarter', 'year']
        :return: start_date_rows, end_date_rows
        """
        if self.start_date is None and self.end_date is None:
            print("\'sesonal_variation_of_()\' need to execute \'get_trend_of_()\' first.")
            return [], []

        dict_duration = {'start_date': self.start_date, 'end_date': self.end_date}



        # print(dict_duration)


        start_date_rows, end_date_rows = jt.SPATable().retrieve_user_variation_tmp(user_id, assess_type, dict_duration)

        if start_date_rows is not () and end_date_rows is not ():
            return start_date_rows, end_date_rows

        else:
            return [], []

    @staticmethod
    def _retrieve_user_trend(user_id, assess_type, dict_duration):
        """
        To return all rows from start_date to end_date

        :param user_id: Integer
        :param assess_type: String, one of ['aging_diagnosis', 'emotion_diagnosis']
        :param dict_duration: Dictionary, {'start_date': (YYYYMMDD), 'end_date': (YYYYMMDD)}
        :return: rows: list of dictionaries
        """

        rows = jt.SPATable().retrieve_user_trend(user_id, assess_type, dict_duration)

        if rows is ():
            return []

        else:
            return rows

    @staticmethod
    def _calculate_age_sum(start_date_rows, end_date_rows):

        dict_start_sum = {}
        dict_end_sum = {}

        for row in start_date_rows:
            for attribute in row:
                if attribute not in ['user_id', 'session_id', 'assess_type', 'result', 'assess_id',
                                     'Age_Wrinkle', 'Age_Spot', 'Age_Geo', 'recorded_date']:
                    if attribute in list(dict_start_sum.keys()):
                        if row[attribute] is None:
                            continue
                        dict_start_sum[attribute] += row[attribute]

                    elif attribute not in list(dict_start_sum.keys()):
                        if row[attribute] is None:
                            continue
                        dict_start_sum[attribute] = row[attribute]

        for row in end_date_rows:
            for attribute in row:
                if attribute not in ['user_id', 'session_id', 'assess_type', 'result', 'assess_id',
                                     'Age_Wrinkle', 'Age_Spot', 'Age_Geo', 'recorded_date']:
                    if attribute in list(dict_end_sum.keys()):
                        if row[attribute] is None:
                            continue
                        dict_end_sum[attribute] += row[attribute]

                    elif attribute not in list(dict_end_sum.keys()):
                        if row[attribute] is None:
                            continue
                        dict_end_sum[attribute] = row[attribute]

        return dict_start_sum, dict_end_sum

    @staticmethod
    def _calculate_emotion_sum(start_date_rows, end_date_rows):

        dict_start_sum = {'f1': 0, 'f2': 0, 'f3': 0, 'f4': 0,
                          'f5': 0, 'f6': 0, 'f7': 0, 'f8': 0,
                          'f9': 0, 'f10': 0, 'f11': 0}

        dict_end_sum = {'f1': 0, 'f2': 0, 'f3': 0, 'f4': 0,
                        'f5': 0, 'f6': 0, 'f7': 0, 'f8': 0,
                        'f9': 0, 'f10': 0, 'f11': 0}

        for i in start_date_rows:
            dict_start_sum['f1'] += i['f1']
            dict_start_sum['f2'] += i['f2']
            dict_start_sum['f3'] += i['f3']
            dict_start_sum['f4'] += i['f4']
            dict_start_sum['f5'] += i['f5']
            dict_start_sum['f6'] += i['f6']
            dict_start_sum['f7'] += i['f7']
            dict_start_sum['f8'] += i['f8']
            dict_start_sum['f9'] += i['f9']
            dict_start_sum['f10'] += i['f10']
            dict_start_sum['f11'] += i['f11']

        for i in end_date_rows:
            dict_end_sum['f1'] += i['f1']
            dict_end_sum['f2'] += i['f2']
            dict_end_sum['f3'] += i['f3']
            dict_end_sum['f4'] += i['f4']
            dict_end_sum['f5'] += i['f5']
            dict_end_sum['f6'] += i['f6']
            dict_end_sum['f7'] += i['f7']
            dict_end_sum['f8'] += i['f8']
            dict_end_sum['f9'] += i['f9']
            dict_end_sum['f10'] += i['f10']
            dict_end_sum['f11'] += i['f11']

            return dict_start_sum,dict_end_sum


class AgingProgressAnalyzer(ProgressAnalyzer):

    assess_type = 'aging_diagnosis'

    def __init__(self):
        self.start_date = None
        self.end_date = None

    def get_seasonal_variation_of_factors(self, user_id, rank=3):
        return self._get_seasonal_variation_of_factors(user_id, self.assess_type, 'spaaf',rank)

    def get_seasonal_variation_of_age(self, user_id):
        """
        :param user_id: Integer
        :param duration: String, one of ['week', 'month', 'quarter', 'year'] or Integer
        :return: state: list of dictionary,
        and dictionary of values are String which has form of ['increasing','17%'],
         and state is one of ['increasing','maintaining','decreasing']
        """
        start_date_rows, end_date_rows = self._retrieve_user_variation_tmp(user_id, self.assess_type)

        # print(start_date_rows)
        # print(end_date_rows)

        # When retrieving methods returns nothing.
        if start_date_rows == [] and end_date_rows == []:
            return {'formal average age': -1, 'latter average age': -1, 'state': ''}

        # n일 전의 평가 나이 총합과 오늘의 평가 나이 총합을 저장
        dict_start_sum = {'age': 0}
        dict_end_sum = {'age': 0}

        for i in start_date_rows:
            dict_start_sum['age'] += int(i['result'])

        for i in end_date_rows:
            dict_end_sum['age'] += int(i['result'])

        #  To calculate each points average of predicted age
        start_avg_age = round(dict_start_sum['age'] / len(start_date_rows), 5)
        end_avg_age = round(dict_end_sum['age'] / len(end_date_rows), 5)

        try:
            result_ratio = round(end_avg_age/start_avg_age, 2)
        except ZeroDivisionError:
            if end_avg_age != 0:
                result_ratio = 2
            else:
                result_ratio = 1

        if result_ratio > 1.05:
            state = ['increasing', f'{round(abs(1 - result_ratio) * 100,0)}%']

        elif 1.05 >= result_ratio >= 0.95:
            state = ['maintaining', '']

        else:
            state = ['decreasing', f'{round(abs(1 - result_ratio) * 100,0)}%']

        return {'formal average age': start_avg_age, 'latter average age': end_avg_age, 'state': state}

    def get_trend_of_age(self, user_id, user_name, dict_duration, interval, dict_fact=None):
        """
        To get two charts of age trend within passed duration
        :param user_id: Integer
        :param user_name: String
        :param dict_duration: Dictionary, {'start_date': (YYYYMMDD), 'end_date': (YYYYMMDD)}
        :param interval : String, one of ['day', 'week', 'month', quarter', 'year']
        :param dict_fact: Dictionary which have one key, and one value for list {'factors':['']}
        :return: String, path of image
        """

        # To check dict_duration format
        for date in dict_duration:
            error_count = 0
            if date not in ['start_date', 'end_date']:
                print(f"Invalid Key: \'{date}\'. : WC.progress analyzer.get_trend_of_age")
                error_count += 1

            elif not isinstance(dict_duration[date], str):
                print(f"Invalid Value Type: \'{type(dict_duration[date])}\'. "
                      f"Value should be String. : WC.progress analyzer.get_trend_of_age")
                error_count += 1
            if error_count > 0:
                return "", ""

        # To check interval format
        try:
            interval = interval.lower()

            if interval == 'daily':
                interval = 'day'
            else:
                interval = interval[:-2]

        except AttributeError as a:
            print(f'Invalid type: \'{type(interval)}\'. '
                  f'\'interval\' should be string. : WC.progress analyzer.get_trend_of_age')
            return "", ""

        if interval not in ['day', 'custom', 'week', 'month', 'quarter', 'year']:
            print(f"Invalid interval: \'{interval}\'. : WC.progress analyzer.get_trend_of_age")
            return "", ""

        # To change dict_duration values into 'datetime' type
        try:
            dict_duration['start_date'] = datetime.strptime(dict_duration['start_date'], '%Y%m%d')

        # when 'start_date' input is missing
        except KeyError:
            print('ERROR: WC.progress analyzer.get_trend_of_age : start_date is missing.')
            return '', ''

        except ValueError:
            try:
                dict_duration['start_date'] = datetime.strptime(dict_duration['start_date'], '%Y-%m-%d')
            except ValueError:
                print(f'Invalid date format: \'{list(dict_duration.values())[1]}\'.'
                      f' Format is \'YYYYMMDD\' or \'YYYY-MM-DD\'')
                return '', ''

        try:
            dict_duration['end_date'] = datetime.strptime(dict_duration['end_date'], '%Y%m%d')

        # when 'end_date' input is missing
        except KeyError:
            print('ERROR: WC.progress analyzer.get_trend_of_age : end_date is missing.')
            return '', ''

        except ValueError:
            try:
                dict_duration['end_date'] = datetime.strptime(dict_duration['end_date'], '%Y-%m-%d')
            except ValueError:
                print(f'Invalid date format: \'{list(dict_duration.values())[1]}\'.'
                      f' Format is \'YYYYMMDD\' or \'YYYY-MM-DD\'')
                return '', ''

        # Pick 'rule' by given option
        # OPTION
        # D : daily
        # W-MON : WEEKLY
        # MS : MONTHLY
        # QS : QUARTERLY
        # AS : ANNUALLY

        if interval == 'day':
            interval = 'D'
        elif interval == 'week':
            interval = 'W-MON'
        elif interval == 'month':
            interval = 'MS'
        elif interval == 'quarter':
            interval = 'QS'
        elif interval == 'year':
            interval = 'AS'

        # seasonal_variation_function will use these variables.
        self.start_date = dict_duration['start_date']
        self.end_date = dict_duration['end_date']

        # To return all rows of predicted age result within passed duration
        age_rows = self._retrieve_user_trend(user_id, self.assess_type, dict_duration)

        if not age_rows:
            print('No age_rows')
            return '', ''

        # To get path of saved stacked bar graph
        bar = self.compute_age_progress(age_rows, user_name, interval, dict_fact)

        # To get path of saved pie chart
        pie_chart = self.compute_age_ratio(age_rows, user_name)

        if bar is not '' and pie_chart is not '':
            return bar, pie_chart

        else:
            print("NO RETURN VALUE : WC.progress analyzer.get_trend_of_age")
            print("This error occurs when one or both of return value are []")
            return '', ''

    def compute_age_progress(self, list_dict_result, user_name, interval, dict_fact=None):
        """
        :param list_dict_result: lisf of dictionararies
        :param user_name: String
        :param interval: String, one of ['D', 'W-MON', 'MS', 'QS', 'AS']
        :param dict_fact:{'factors':['', '', '' ...]}
        :return:
        """
        # if dict_fact is empty, methods returns graph presenting whole factors
        if dict_fact is None:
            dict_fact = {'factors': ['Age', 'Age_Wrinkle', 'Age_Spot', 'Age_Geo', 'Appearance']}

        df = pd.DataFrame(list_dict_result)

        # print('phase1:\n', df)

        # Code to make mean() method operate
        result = {'Result': list(pd.to_numeric(df['result']).replace(np.nan, 0)),
                  'Age_Wrinkle': list(pd.to_numeric(df['Age_Wrinkle']).replace(np.nan,0)),
                  'Age_Spot': list(pd.to_numeric(df['Age_Spot']).replace(np.nan, 0)),
                  'Age_Geo': list(pd.to_numeric(df['Age_Geo']).replace(np.nan, 0))}

        df = pd.DataFrame(result, index=df['recorded_date'])

        # To resample by given interval
        df = df.resample(rule=f'{interval}').mean()

        # print('phase2:\n', df)
        #
        # result = {'Result': list(pd.to_numeric(df['Result']).replace(np.nan, 0)),
        #           'Age_Wrinkle': list(pd.to_numeric(df['Age_Wrinkle']).replace(np.nan,0)),
        #           'Age_Spot': list(pd.to_numeric(df['Age_Spot']).replace(np.nan, 0)),
        #           'Age_Geo': list(pd.to_numeric(df['Age_Geo']).replace(np.nan, 0))}
        #
        # df = pd.DataFrame(result, index=df.index)
        #
        # print('phase3:\n', df)

        x = [datetime.strftime(i, "%Y-%m-%d") for i in df.index]

        # Code to change format of Date
        if interval == 'MS':
            x = [i[:7] for i in x]

        elif interval == 'QS':
            for i in range(len(x)):
                # print(x[i][5:7])
                if x[i][5:7] == '01':
                    x[i] = x[i][0:5] + ' 1Q'
                elif x[i][5:7] == '04':
                    x[i] = x[i][0:5] + ' 2Q'
                elif x[i][5:7] == '07':
                    x[i] = x[i][0:5] + ' 3Q'
                elif x[i][5:7] == '10':
                    x[i] = x[i][0:5] + ' 4Q'

        elif interval == 'AS':
            x = [i[:4] for i in x]

        appearance = [i for i in df['Result']]
        age_wrinkle = [i for i in df['Age_Wrinkle']]
        age_spot = [i for i in df['Age_Spot']]
        age_geo = [i for i in df['Age_Geo']]

        # appearance = [int(i) for i in df['Result']]
        # age_wrinkle = [int(i) for i in df['Age_Wrinkle']]
        # age_spot = [int(i) for i in df['Age_Spot']]
        # age_geo = [int(i) for i in df['Age_Geo']]

        # Code to get average age from entire model's result [493:517]
        age = []
        for i in range(0, len(appearance)):
            divisor = 4
            dividend = []

            if appearance[i] == 0:
                divisor -= 1
            else:
                dividend.append(appearance[i])
            if age_wrinkle[i] == 0:
                divisor -= 1
            else:
                dividend.append(age_wrinkle[i])
            if age_spot[i] == 0:
                divisor -= 1
            else:
                dividend.append(age_spot[i])
            if age_geo[i] == 0:
                divisor -= 1
            else:
                dividend.append(age_geo[i])

            try:
                age.append(sum(i for i in dividend)/divisor)
            # When all rows are zero
            except ZeroDivisionError:
                age.append(0)

        #
        factors = dict_fact['factors']

        if 'Age' in factors:
            plt.plot(x, age, 'o-', color='green', label='Age', zorder=10)
        if 'Age_Wrinkle' in factors:
            plt.plot(x, age_wrinkle, 'o-', color='darkgreen', label='Wrinkle', zorder=10)
        if 'Age_Spot' in factors:
            plt.plot(x, age_spot, 'o-', color='orange', label='Spot', zorder=10)
        if 'Age_Geo' in factors:
            plt.plot(x, age_geo, 'o-', color='purple', label='Geo(dis,shp)', zorder=10)
        if 'Appearance' in factors:
            plt.plot(x, appearance, 'o-', color='darkblue', label='Appearance', zorder=10)

        plt.grid(True, color='gray', linestyle='dashed', linewidth=0.5, zorder=5)

        # Code to rotate x label's name to prevent overlapping
        if len(x) > 5:
            plt.xticks(x, rotation=90)

        plt.ylabel('Predicted Age')

        plt.legend().set_zorder(11)

        if x[0] != x[-1]:
            plt.title(f'{user_name} \'s Age Progression\n'
                      f'\'{x[0]} ~ {x[-1]}\'')
        else:
            plt.title(f'{user_name} \'s Age Progression\n')

        # make ndarray part
        save_path = f"./saved_chart_image/age_line.png"
        plt.savefig(save_path)

        pie_chart = cv2.imread(save_path)

        if pie_chart.any():
            plt.close()
            return save_path

        else:
            print("ERROR: WC.progressanalyzer.compute_emotion_frequency : histogram get None")
            return []

    def compute_age_ratio(self, list_dict_result, user_name):
        """
        To return pie chart
        :param list_dict_result: List
        :param user_id: Integer
        :param duration: String, one of ['week', 'month', 'quarter', 'year']
        :return: pie_chart : ndarray
        """
        start_date = datetime.strftime(self.start_date, '%Y-%m-%d')
        end_date = datetime.strftime(self.end_date, '%Y-%m-%d')

        # To get list of all predicted age
        list_age = []
        for i in list_dict_result:
            list_age.append(int(i['result']))

        # To get average age
        age_sum = 0
        for i in list_age:
            age_sum += i
        avg_age = age_sum/len(list_age)

        # To get dictionary of frequency of each age
        dict_sum = dict()
        for i in list_age:
            if str(i) in dict_sum.keys():
                dict_sum[str(i)] += 1
            else:
                dict_sum[str(i)] = 1

        # print(dict_sum)

        # even if sorted list's length is under 5, it doesn't raise exception
        sorted_age_sum = sorted(dict_sum.items(), key=operator.itemgetter(1), reverse=True)[0:5]

        top5_age = [i[0] for i in sorted_age_sum]

        organized_age_list = {'etc': 0}

        for i in dict_sum:
            if i not in top5_age:
                organized_age_list['etc'] += dict_sum[i]
            else:
                organized_age_list[i] = dict_sum[i]

        # print(organized_age_list)

        # Code to save ratio for each emotion into list in order
        try:
            ratio = [round(organized_age_list[age]/len(list_dict_result)*100, 1) for age in organized_age_list]
        except ZeroDivisionError:
            print("list_dict_result is [].")
            print("ERROR: WC.progress analyzer.compute_age_ratio")
            return []

        labels = [i for i in organized_age_list]
        plt.pie(ratio, labels=labels, explode=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
                startangle=90, autopct='%.1f%%', pctdistance=0.8)
        plt.title(f'{user_name} \'s Predicted Age Ratio\n'
                  f'\'{start_date} ~ {end_date}\'')

        # make nd array part
        save_path = f"./saved_chart_image/age_pie_chart.png"

        # To make pie chart looks like doughnut
        plt.gca().add_artist(plt.Circle((0, 0), 0.70, color='black', fc='white', linewidth=0))
        plt.gca().annotate(f'Average Age : \'{round(avg_age,1)}\'', xy=(0, 0), ha="center", fontsize=11, weight='bold')

        plt.savefig(save_path)

        pie_chart = cv2.imread(save_path)

        if pie_chart.any():
            plt.close()
            return save_path
        else:
            print("ERROR: WC.progressanalyzer.compute_age_ratio : pie_chart get None")
            return []


class EmotionProgressAnalyzer(ProgressAnalyzer):

    assess_type = 'emotion_diagnosis'

    def __init__(self):
        self.start_date = None
        self.end_date = None

    def get_seasonal_variation_of_factors(self, user_id, duration, rank=3):
        return self._get_seasonal_variation_of_factors(user_id, self.assess_type, 'spaef', rank)

    def get_seasonal_variation_of_emotions(self, user_id, rank=3):
        """
        :param user_id: Integer
        :param duration: String, one of ['week', 'month', 'quarter', 'year'] or Integer
        :param rank: Integer, (1<=rank<=10)
        :return: state: list of dictionary,
        and dictionary of values are String which has form of ['increasing','17%'],
         and state is one of ['increasing','maintaining','decreasing']
        """

        # There are 10 factors for detecting emotions
        if rank < 1 or rank > 10:
            print("invalid rank value.")
            return ['', '']

        start_date_rows, end_date_rows = self._retrieve_user_variation_tmp(user_id, self.assess_type)

        # When retrieving methods returns nothing.
        if start_date_rows == [] and end_date_rows == []:
            return ['', '']

        # Count and save frequency of predicted emotions
        # 'Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'

        dict_start_sum = {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0,
                          'sad': 0, 'surprise': 0, 'neutral': 0}

        dict_end_sum = {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0,
                        'sad': 0, 'surprise': 0, 'neutral': 0}

        # count detected emotion and save into dictionary in start_date
        for i in start_date_rows:
            emotion = i['result']
            if emotion == '0':
                dict_start_sum['angry'] += 1
            elif emotion is '1':
                dict_start_sum['disgust'] += 1
            elif emotion is '2':
                dict_start_sum['fear'] += 1
            elif emotion is '3':
                dict_start_sum['happy'] += 1
            elif emotion is '4':
                dict_start_sum['sad'] += 1
            elif emotion is '5':
                dict_start_sum['surprise'] += 1
            elif emotion is '6':
                dict_start_sum['neutral'] += 1

        # count detected emotion and save into dictionary in end_date
        for i in end_date_rows:
            emotion = i['result']
            if emotion is '0':
                dict_end_sum['angry'] += 1
            elif emotion is '1':
                dict_end_sum['disgust'] += 1
            elif emotion is '2':
                dict_end_sum['fear'] += 1
            elif emotion is '3':
                dict_end_sum['happy'] += 1
            elif emotion is '4':
                dict_end_sum['sad'] += 1
            elif emotion is '5':
                dict_end_sum['surprise'] += 1
            elif emotion is '6':
                dict_end_sum['neutral'] += 1

        # 상위로부터 'rank' 개 나열
        start_sorted_emotion = sorted(dict_start_sum.items(), key=operator.itemgetter(1), reverse=True)[0:rank]
        # print(dict_start_sum)
        # print(dict_end_sum)
        # print(start_sorted_emotion)
        result = []

        # To make result list
        for i in range(0, rank):

            emotion_name = start_sorted_emotion[i][0]

            start_emotion_ratio = round(dict_start_sum[f'{emotion_name}'] / len(start_date_rows), 5)
            end_emotion_ratio = round(dict_end_sum[f'{emotion_name}'] / len(end_date_rows), 5)

            try:
                result_ratio = round(end_emotion_ratio / start_emotion_ratio, 2)

            except ZeroDivisionError:
                if end_emotion_ratio != 0:
                    result_ratio = 2
                else:
                    result_ratio = 1

            if result_ratio > 1.05:
                state = ['increasing', f'{round(abs(1 - result_ratio) * 100,0)}%']

            elif 1.05 >= result_ratio >= 0.95:
                state = ['maintaining', '']

            else:
                state = ['decreasing', f'{round(abs(1 - result_ratio) * 100,0)}%']

            result.append({f'{emotion_name}': state})

        return result

    # def get_top_emotions(self, user_id, duration):
    #     """
    #     To get top emotions from each two points(start, end)
    #     :param user_id: Integer
    #     :param duration: String, one of ['week', 'month', 'quarter', 'year'] or Integer
    #     :return: result : ex) {"start_top_emotion" : 'angry', "end_top_emotion" : 'angry'}
    #     """
    #
    #     start_date_rows, end_date_rows = self._retrieve_user_variation_tmp(user_id, self.assess_type, duration)
    #
    #     # When retrieving methods returns nothing.
    #     if start_date_rows == [] and end_date_rows == []:
    #         return {"start_top_emotion": '', "end_top_emotion": ''}
    #
    #     # Count and save frequency of predicted emotions
    #     # 'Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'
    #
    #     dict_start_sum = {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0,
    #                       'sad': 0, 'surprise': 0, 'neutral': 0}
    #
    #     dict_end_sum = {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0,
    #                     'sad': 0, 'surprise': 0, 'neutral': 0}
    #
    #     # count detected emotion and save into dictionary in start_date
    #     for i in start_date_rows:
    #         emotion = i['result']
    #         if emotion == '0':
    #             dict_start_sum['angry'] += 1
    #         elif emotion is '1':
    #             dict_start_sum['disgust'] += 1
    #         elif emotion is '2':
    #             dict_start_sum['fear'] += 1
    #         elif emotion is '3':
    #             dict_start_sum['happy'] += 1
    #         elif emotion is '4':
    #             dict_start_sum['sad'] += 1
    #         elif emotion is '5':
    #             dict_start_sum['surprise'] += 1
    #         elif emotion is '6':
    #             dict_start_sum['neutral'] += 1
    #
    #     # count detected emotion and save into dictionary in end_date
    #     for i in end_date_rows:
    #         emotion = i['result']
    #         if emotion is '0':
    #             dict_end_sum['angry'] += 1
    #         elif emotion is '1':
    #             dict_end_sum['disgust'] += 1
    #         elif emotion is '2':
    #             dict_end_sum['fear'] += 1
    #         elif emotion is '3':
    #             dict_end_sum['happy'] += 1
    #         elif emotion is '4':
    #             dict_end_sum['sad'] += 1
    #         elif emotion is '5':
    #             dict_end_sum['surprise'] += 1
    #         elif emotion is '6':
    #             dict_end_sum['neutral'] += 1
    #
    #     start_sorted_emotion = sorted(dict_start_sum.items(), key=operator.itemgetter(1), reverse=True)[0:1]
    #     end_sorted_emotion = sorted(dict_end_sum.items(), key=operator.itemgetter(1), reverse=True)[0:1]
    #
    #     start_top_emotion = start_sorted_emotion[0][0]
    #     end_top_emotion = end_sorted_emotion[0][0]
    #
    #     result = {"start_top_emotion": start_top_emotion, "end_top_emotion": end_top_emotion}
    #
    #     return result

    def get_trend_of_emotions(self, user_id, user_name, dict_duration, interval, dict_fact=None):
        """
        To get two charts of emtion trend within passed duration
        :param user_id: Integer
        :param user_name: String
        :param dict_duration: Dictionary, {'start_date': (YYYYMMDD), 'end_date': (YYYYMMDD)}
        :param interval : String, one of ['day', 'week', 'month', quarter', 'year']
        :return: String, path of image
        """

        # To check dict_duration format
        for date in dict_duration:
            error_count = 0
            if date not in ['start_date', 'end_date']:
                print(f"Invalid Key: \'{date}\'. : WC.progress analyzer.get_trend_of_emotion")
                error_count += 1

            elif not isinstance(dict_duration[date], str):
                print(f"Invalid Value Type: \'{type(dict_duration[date])}\'. "
                      f"Value should be String. : WC.progress analyzer.get_trend_of_emotion")
                error_count += 1
            if error_count > 0:
                return "", ""

        # To check interval format
        try:
            interval = interval.lower()

            if interval == 'daily':
                interval = 'day'
            else:
                interval = interval[:-2]

        except AttributeError as a:
            print(f'Invalid type: \'{type(interval)}\'. '
                  f'\'interval\' should be string. : WC.progress analyzer.get_trend_of_emotion')
            return "", ""

        if interval not in ['day', 'week', 'month', 'quarter', 'year']:
            print(f"Invalid interval: \'{interval}\'. : WC.progress analyzer.get_trend_of_emotion")
            return "", ""

        # To change dict_duration values into 'datetime' type
        try:
            dict_duration['start_date'] = datetime.strptime(dict_duration['start_date'], '%Y%m%d')

        # when 'start_date' input is missing
        except KeyError:
            if interval in ['week', 'month', 'quarter', 'year']:
                pass
            else:
                print('Please choose interval type. ERROR: WC.progression analyzer.get_trend_of_emotion')
                return "", ""

        except ValueError:
            try:
                dict_duration['start_date'] = datetime.strptime(dict_duration['start_date'], '%Y-%m-%d')
            except ValueError:
                print(f'Invalid date format: \'{list(dict_duration.values())[1]}\'.'
                      f' Format is \'YYYYMMDD\' or \'YYYY-MM-DD\'')
                return '', ''


        try:
            dict_duration['end_date'] = datetime.strptime(dict_duration['end_date'], '%Y%m%d')

        # when 'end_date' input is missing
        except KeyError:
            if interval in ['week', 'month', 'quarter', 'year']:
                pass
            else:
                print('Please choose interval type. ERROR: WC.progression analyzer.get_trend_of_emotion')
                return "", ""

        except ValueError:
            try:
                dict_duration['end_date'] = datetime.strptime(dict_duration['end_date'], '%Y-%m-%d')
            except ValueError:
                print(f'Invalid date format: \'{list(dict_duration.values())[1]}\'.'
                      f' Format is \'YYYYMMDD\' or \'YYYY-MM-DD\'')
                return '', ''

        if interval == 'day':
            interval = 'D'
        elif interval == 'week':
            interval = 'W-MON'
        elif interval == 'month':
            interval = 'MS'
        elif interval == 'quarter':
            interval = 'QS'
        elif interval == 'year':
            interval = 'AS'

        # print(interval)

        self.start_date = dict_duration['start_date']
        self.end_date = dict_duration['end_date']

        # To return all rows of predicted emotion result within passed duration
        emotion_rows = self._retrieve_user_trend(user_id, self.assess_type, dict_duration)

        if not emotion_rows:
            print('No emotion_rows')
            return '', ''

        # To get path of saved histogram graph
        histogram = self.compute_emotion_frequency(emotion_rows, user_name, interval, dict_fact)

        # To get path of saved pie chart
        pie_chart = self.compute_emotion_progress(emotion_rows, user_name, interval, dict_fact)

        if histogram is not '' and pie_chart is not '':
            return histogram, pie_chart
        else:
            print("NO RETURN VALUE : WC.progress analyzer.get_trend_of_emotion")
            print("This error occurs when one or both of return value are []")
            return '', ''

    def compute_emotion_frequency(self, list_dict_result, user_name, interval, dict_fact=None):
        """
        :param list_dict_result:
        :param user_name:
        :param interval:
        :param dict_fact:
        :return:
        """

        if dict_fact is None:
            dict_fact = {'factors': ['Emotion', 'Geo']}

        # print(list_dict_result)

        df = pd.DataFrame(list_dict_result)
        result = {'Result': list(pd.to_numeric(df['result']).replace(np.nan, 0)),
                  'Factor_Result': list(pd.to_numeric(df['factor_result']).replace(np.nan, 0))}

        df = pd.DataFrame(result, index=df['recorded_date'])

        # To make graph, need to make new data frame
        # To get result's average
        ratio_results = {}
        # To get factor result's average
        ratio_factor_results = {}

        # Daily
        if interval == 'D':
            # To get number of x axis
            num_xticks = (self.end_date - self.start_date).days + 1  # '+ 1' : To include end_date

            # num_xticks 만큼 반복
            date = self.start_date

            for i in range(0, num_xticks):

                start = date
                end = date + timedelta(days=1)
                result = [0, 0, 0, 0, 0, 0, 0]

                # 조건을 하루씩 하여 해당 행을 가져오는 코드
                df_tmp = df.loc[(df.index >= start) & (df.index < end)]

                # Empty DataFrame
                if len(df_tmp.index) == 0:
                    # ratio_results[datetime.strftime(date, '%Y-%m-%d')] = result
                    # ratio_factor_results[datetime.strftime(date, '%Y-%m-%d')] = result
                    date += timedelta(days=1)
                    continue

                # Not Empty DataFrame

                for i in df_tmp['Result']:
                    if i == 0:
                        result[0] += 1
                    elif i == 1:
                        result[1] += 1
                    elif i == 2:
                        result[2] += 1
                    elif i == 3:
                        result[3] += 1
                    elif i == 4:
                        result[4] += 1
                    elif i == 5:
                        result[5] += 1
                    elif i == 6:
                        result[6] += 1

                for i in range(0, len(result)):
                    result[i] = round(result[i]/len(df_tmp.index) * 100, 0)
                ratio_results[datetime.strftime(date, '%Y-%m-%d')] = result

                result = [0, 0, 0, 0, 0, 0, 0]

                for i in df_tmp['Factor_Result']:
                    if i == 0:
                        result[0] += 1
                    elif i == 1:
                        result[1] += 1
                    elif i == 2:
                        result[2] += 1
                    elif i == 3:
                        result[3] += 1
                    elif i == 4:
                        result[4] += 1
                    elif i == 5:
                        result[5] += 1
                    elif i == 6:
                        result[6] += 1

                for i in range(0, len(result)):
                    result[i] = round(result[i]/len(df_tmp.index) * 100, 0)
                ratio_factor_results[datetime.strftime(date, '%Y-%m-%d')] = result

                date += timedelta(days=1)

        elif interval == 'W-MON':

            # To get number of x axis
            # num_xticks 만큼 반복

            num_xticks = ceil((self.end_date - self.start_date).days / 7)  # '+ 1' : To include end_date

            date = self.start_date

            for i in range(0, num_xticks):
                start = date
                end = date + relativedelta(weeks=1)
                result = [0, 0, 0, 0, 0, 0, 0]

                # 조건을 하루씩 하여 해당 행을 가져오는 코드
                df_tmp = df.loc[(df.index >= start) & (df.index < end)]

                # Empty DataFrame
                if len(df_tmp.index) == 0:
                    # ratio_results[datetime.strftime(date, '%Y-%m-%d')] = result
                    # ratio_factor_results[datetime.strftime(date, '%Y-%m-%d')] = result
                    date += timedelta(weeks=1)
                    continue

                # Not Empty DataFrame
                for i in range(len(df_tmp['Result'])):
                    for j in range(7):
                        if df_tmp['Result'][i] == j:
                            result[j] +=1

                for i in range(0, len(result)):
                    result[i] = round(result[i]/len(df_tmp.index) * 100, 0)

                ratio_results[datetime.strftime(date, '%Y-%m-%d')] = result

                result = [0, 0, 0, 0, 0, 0, 0]

                for i in range(len(df_tmp['Factor_Result'])):
                    for j in range(7):
                        if df_tmp['Factor_Result'][i] == j:
                            result[j] += 1

                for i in range(0, len(result)):
                    result[i] = round(result[i]/len(df_tmp.index) * 100, 0)

                ratio_factor_results[datetime.strftime(date, '%Y-%m-%d')] = result

                date += timedelta(weeks=1)

        elif interval == 'MS':

            # To get number of x axis
            # num_xticks 만큼 반복

            num_xticks = self.end_date.month - self.start_date.month + 1

            date = self.start_date

            for i in range(0, num_xticks):
                start = datetime.strptime(f'{date.year}-{date.month}', '%Y-%m')
                end = datetime.strptime(f'{date.year}-{date.month+1}', '%Y-%m')
                result = [0, 0, 0, 0, 0, 0, 0]


                # 조건을 하루씩 하여 해당 행을 가져오는 코드
                df_tmp = df.loc[(df.index.month >= start.month) & (df.index < end)]

                # Empty DataFrame
                if len(df_tmp.index) == 0:
                    # ratio_results[datetime.strftime(date, '%Y-%m-%d')] = result
                    # ratio_factor_results[datetime.strftime(date, '%Y-%m-%d')] = result
                    date += relativedelta(months=1)
                    continue

                # Not Empty DataFrame
                for i in range(len(df_tmp['Result'])):
                    for j in range(7):
                        if df_tmp['Result'][i] == j:
                            result[j] +=1

                for i in range(0, len(result)):
                    result[i] = round(result[i]/len(df_tmp.index) * 100, 0)

                ratio_results[datetime.strftime(date, '%Y-%m-%d')] = result

                result = [0, 0, 0, 0, 0, 0, 0]

                for i in range(len(df_tmp['Factor_Result'])):
                    for j in range(7):
                        if df_tmp['Factor_Result'][i] == j:
                            result[j] += 1

                for i in range(0, len(result)):
                    result[i] = round(result[i]/len(df_tmp.index) * 100, 0)

                ratio_factor_results[datetime.strftime(date, '%Y-%m-%d')] = result

                date += relativedelta(months=1)

        elif interval == 'QS':

            # To get number of x axis
            # num_xticks 만큼 반복

            num_xticks = self.end_date.year - self.start_date.year + 1

            date = self.start_date

            for i in range(0, num_xticks):
                start = datetime.strptime(f'{date.year}', '%Y')
                end = datetime.strptime(f'{date.year+1}', '%Y')
                result = [0, 0, 0, 0, 0, 0, 0]

                # 조건을 하루씩 하여 해당 행을 가져오는 코드
                df_tmp = df.loc[(df.index >= start) & (df.index < end)]

                # Empty DataFrame
                if len(df_tmp.index) == 0:
                    # ratio_results[datetime.strftime(date, '%Y-%m-%d')] = result
                    # ratio_factor_results[datetime.strftime(date, '%Y-%m-%d')] = result
                    date += relativedelta(years=1)
                    continue

                # Not Empty DataFrame
                for i in range(len(df_tmp['Result'])):
                    for j in range(7):
                        if df_tmp['Result'][i] == j:
                            result[j] +=1

                for i in range(0, len(result)):
                    result[i] = round(result[i]/len(df_tmp.index) * 100, 0)

                ratio_results[datetime.strftime(date, '%Y-%m-%d')] = result

                result = [0, 0, 0, 0, 0, 0, 0]

                for i in range(len(df_tmp['Factor_Result'])):
                    for j in range(7):
                        if df_tmp['Factor_Result'][i] == j:
                            result[j] += 1

                for i in range(0, len(result)):
                    result[i] = round(result[i]/len(df_tmp.index) * 100, 0)

                ratio_factor_results[datetime.strftime(date, '%Y-%m-%d')] = result

                date += relativedelta(years=1)

        elif interval == 'AS':

            # To get number of x axis
            # num_xticks 만큼 반복

            num_xticks = self.end_date.year - self.start_date.year + 1

            date = self.start_date

            for i in range(0, num_xticks):
                start = datetime.strptime(f'{date.year}', '%Y')
                end = datetime.strptime(f'{date.year+1}', '%Y')
                result = [0, 0, 0, 0, 0, 0, 0]

                # 조건을 하루씩 하여 해당 행을 가져오는 코드
                df_tmp = df.loc[(df.index >= start) & (df.index < end)]

                # Empty DataFrame
                if len(df_tmp.index) == 0:
                    # ratio_results[datetime.strftime(date, '%Y-%m-%d')] = result
                    # ratio_factor_results[datetime.strftime(date, '%Y-%m-%d')] = result
                    date += relativedelta(years=1)
                    continue

                # Not Empty DataFrame
                for i in range(len(df_tmp['Result'])):
                    for j in range(7):
                        if df_tmp['Result'][i] == j:
                            result[j] +=1

                for i in range(0, len(result)):
                    result[i] = round(result[i]/len(df_tmp.index) * 100, 0)

                ratio_results[datetime.strftime(date, '%Y-%m-%d')] = result

                result = [0, 0, 0, 0, 0, 0, 0]

                for i in range(len(df_tmp['Factor_Result'])):
                    for j in range(7):
                        if df_tmp['Factor_Result'][i] == j:
                            result[j] += 1

                for i in range(0, len(result)):
                    result[i] = round(result[i]/len(df_tmp.index) * 100, 0)

                ratio_factor_results[datetime.strftime(date, '%Y-%m-%d')] = result

                date += relativedelta(years=1)

        category_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

        # Organize X labels
        labels = list(ratio_results.keys())

        if interval == 'MS':
            labels = [i[:7] for i in labels]

        elif interval == 'QS':
            for i in range(len(labels)):
                if labels[i][5:7] == '01':
                    labels[i] = labels[i][0:5] +' 1Q'
                elif labels[i][5:7] == '04':
                    labels[i] = labels[i][0:5] +' 2Q'
                elif labels[i][5:7] == '07':
                    labels[i] = labels[i][0:5] +' 3Q'
                elif labels[i][5:7] == '10':
                    labels[i] = labels[i][0:5] +' 4Q'

        elif interval == 'AS':
            labels = [i[:4] for i in labels]

        data = np.array(list(ratio_results.values()))
        data_cum = data.cumsum(axis=1)
        category_colors = plt.get_cmap('RdYlGn')(
            np.linspace(0.15, 0.85, data.shape[1]))

        fig, ax = plt.subplots(figsize=(9.2, 5))

        for i, (colname, color) in enumerate(zip(category_names, category_colors)):
            emotion_rate = data[:, i]
            starts = data_cum[:, i] - emotion_rate
            ax.bar(labels, emotion_rate, bottom =starts, width=0.3,
                    label=colname, color=color)

            xcenters = starts + emotion_rate / 2

            r, g, b, _ = color
            text_color = 'white' if r * g * b < 0.5 else 'darkgrey'

            for y, (x, c) in enumerate(zip(xcenters, emotion_rate)):
                if int(c) == 0:
                    continue
                ax.text(y, x, str(int(c))+'%', ha='center', va='center',
                        color=text_color)

        ax.legend(bbox_to_anchor=(1, 0),
                  loc='lower left', fontsize='small')

        if labels[0] != labels[-1]:
            plt.title(f'{user_name} \'s Emotion Progression\n'
                      f'\'{labels[0]} ~ {labels[-1]}\'')
        else:
            plt.title(f'{user_name} \'s Emotion Progression\n')

        # make ndarray part
        save_path = f"./saved_chart_image/emotion_bar.png"
        plt.savefig(save_path)

        pie_chart = cv2.imread(save_path)

        if pie_chart.any():
            plt.close()
            return save_path

        else:
            print("ERROR: WC.progressanalyzer.compute_emotion_frequency : histogram get None")
            return []

    def compute_emotion_progress(self, list_dict_result, user_name, interval, dict_fact=None):
        """
        :param list_dict_result:
        :param user_name:
        :param interval:
        :param dict_fact:
        :return:
        """
        # Change this code if factor names are changed
        factor_list = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11']

        if dict_fact is None:
            dict_fact = {'factors': [i for i in factor_list]}

        # print(dict_fact)

        start_date = datetime.strftime(self.start_date, '%Y-%m-%d')
        end_date = datetime.strftime(self.end_date, '%Y-%m-%d')

        df = pd.DataFrame(list_dict_result)

        # Code to make mean() method operate
        result = {factor_list[i]: list(pd.to_numeric(df[factor_list[i]]).replace(np.nan, 0))
                  for i in range(0, len(factor_list))}

        df = pd.DataFrame(result, index=df['recorded_date'])

        # To get mean value of each duration

        # Available options
        # D : daily
        # W-MON : WEEKLY
        # MS : MONTHLY
        # QS : QUARTERLY
        # AS : ANNUALLY


        df = df.resample(rule=f'{interval}').mean()

        print(df)

        x = [datetime.strftime(i, "%Y-%m-%d") for i in df.index]

        # Code to change format of Date
        if interval == 'MS':
            x = [i[:7] for i in x]

        elif interval == 'QS':
            for i in range(len(x)):
                print(x[i][5:7])
                if x[i][5:7] == '01':
                    x[i] = x[i][0:5] +' 1Q'
                elif x[i][5:7] == '04':
                    x[i] = x[i][0:5] +' 2Q'
                elif x[i][5:7] == '07':
                    x[i] = x[i][0:5] +' 3Q'
                elif x[i][5:7] == '10':
                    x[i] = x[i][0:5] +' 4Q'
        elif interval == 'AS':
            x = [i[:4] for i in x]

        for i in range(0, len(factor_list)):
            if factor_list[i] in dict_fact['factors']:
                plt.plot(x, [round(i, 2) for i in df[factor_list[i]]], 'o-', label=factor_list[i], zorder=10)

        plt.grid(True, color='gray', linestyle='dashed', linewidth=0.5, zorder=5)

        # Code to rotate x label's name to prevent overlapping

        if len(x) > 5:
            plt.xticks(x, rotation=90)

        # plt.yticks(range(min(y)-5, max(y)+6))

        plt.ylabel('Factor Rate')
        # plt.xlabel('Predicted Age')

        plt.legend(bbox_to_anchor=(1, 0),
                  loc='lower left', fontsize='small')

        if x[0] != x[-1]:
            plt.title(f'{user_name} \'s Emotion Factor Progression\n'
                      f'\'{x[0]} ~ {x[-1]}\'')
        else:
            plt.title(f'{user_name} \'s Emotion Factor Progression\n')

        # make ndarray part
        save_path = f"./saved_chart_image/emotion_line.png"
        plt.savefig(save_path)

        pie_chart = cv2.imread(save_path)

        if pie_chart.any():
            plt.close()
            return save_path

        else:
            print("ERROR: WC.progressanalyzer.compute_emotion_frequency : histogram get None")
            return []

    def compute_emotion_ratio(self, list_dict_result, user_id, user_name):
        """
        To return pie chart
        :param list_dict_result: List
        :param user_id: Integer
        :param duration : String, one of ['week', 'month', 'quarter', 'year']
        :return: pie_chart : ndarray
        """

        start_date = datetime.strftime(self.start_date, '%Y-%m-%d')
        end_date = datetime.strftime(self.end_date, '%Y-%m-%d')

        # duration 동안 총 빈도수를 구하는 코드
        dict_sum = {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0}

        for i in list_dict_result:
            emotion = i['result']
            if emotion == '0':
                dict_sum['angry'] += 1
            elif emotion == '1':
                dict_sum['disgust'] += 1
            elif emotion == '2':
                dict_sum['fear'] += 1
            elif emotion == '3':
                dict_sum['happy'] += 1
            elif emotion == '4':
                dict_sum['sad'] += 1
            elif emotion == '5':
                dict_sum['surprise'] += 1
            elif emotion == '6':
                dict_sum['neutral'] += 1

        # Code to save ratio for each emotion into list in order
        ratio = [round(dict_sum[emotion]/len(list_dict_result), 2)*100 for emotion in dict_sum if dict_sum[emotion] != 0]

        labels = list(i for i in dict_sum if dict_sum[i] != 0)

        plt.pie(ratio, labels=labels, explode=tuple(0.1 for i in labels),
                startangle=90, autopct='%.1f%%', pctdistance=0.8)

        # To make pie chart looks like doughnut
        plt.gca().add_artist(plt.Circle((0, 0), 0.70, color='black', fc='white', linewidth=0))

        plt.title(f'{user_name} \'s Emotion Ratio\n'
                  f'\'{start_date} ~ {end_date}\'')

        # make nd array part
        save_path = f"./saved_chart_image/{user_id}_emotion_pie_chart.png"
        plt.savefig(save_path)

        pie_chart =cv2.imread(save_path)

        if pie_chart.any():
            plt.close()
            return save_path
        else:
            print("ERROR: WC.progressanalyzer.compute_emotion_ratio : pie_chart get None")
            return []


if __name__ == '__main__':

    apa =AgingProgressAnalyzer()

    epa = EmotionProgressAnalyzer()

    # apa.compute_age_frequency(list_dict_result='', user_id=10, duration='week')
    # apa.compute_age_ratio(list_dict_result='', user_id=10, duration='week')

    # print(apa.get_seasonal_variation_of_factors(2, 1))


    # print(apa.get_seasonal_variation_of_age(294, 'week'))
    # print(apa.get_seasonal_variation_of_age(294, 6))


    # apa.get_trend_of_age(user_id=2, user_name='jisoo', dict_duration={'start_date': '20190720', 'end_date': '20190731'})
    # print(apa.get_seasonal_variation_of_age(2))
    # print(apa.get_seasonal_variation_of_factors(2, 1))

    # epa.get_trend_of_emotions(user_id=2, user_name='jisoo', dict_duration={'start_date': '20190720', 'end_date': '20190731'})
    # print(epa.get_seasonal_variation_of_emotions(2))
    # print(epa.get_seasonal_variation_of_factors(2, 1))

    # print(epa.get_seasonal_variation_of_factors(294, 'week'))
    # print(epa.get_seasonal_variation_of_emotions(294, 'week'))
    # print(epa.get_seasonal_variation_of_emotions(294, 6))
    # print(epa.get_seasonal_variation_of_emotions(2, 7))
    # print(epa.get_top_emotions(3, 8))

    # print(type(apa.get_trend_of_age(user_id=2, user_name='jisoo', duration=6)))
    # apa.get_trend_of_age(user_id=2, user_name='jisoo')
    # apa.get_trend_of_age(user_id=2, user_name='jisoo', dict_duration={'start_date': '20190701'}, interval='week')
    # apa.get_trend_of_age(user_id=2, user_name='jisoo', dict_duration={'start_date': '20190720', 'end_date': '20190731'},
    #                      interval='day')
    # apa.get_trend_of_age(user_id=2, user_name='jisoo', dict_duration={'end_date': '20190725'})

    apa.get_trend_of_age(user_id=2, user_name='jisoo', interval='month',
                              dict_duration={'start_date': '20190723', 'end_date': '20190815'})

    # epa.get_trend_of_emotions(user_id=2, user_name='jisoo', interval= 'day',
    #                           dict_duration={'start_date': '20190723', 'end_date': '20190815'})

    # epa.get_seasonal_variation_of_emotions(294, 'week')
    # a, b = epa._retrieve_user_variation(294, 'aging_diagnosis', 'week')

    # print (a)
    # print (b)


