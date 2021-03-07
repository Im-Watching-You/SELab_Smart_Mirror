import WC.joinedtable as jt

from datetime import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta

import matplotlib.pyplot as plt

import random

import cv2

import operator
import numpy as np


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

        # print(dict_start_sum)
        # print(dict_end_sum)

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
            state = ['maintaining','']

        else:
            state = ['decreasing', f'{round(abs(1 - result_ratio) * 100,0)}%']

        return {'formal average age': start_avg_age, 'latter average age': end_avg_age, 'state': state}

    def get_trend_of_age(self, user_id, user_name, dict_duration, interval='custom'):
        """
        To get two charts of age trend within passed duration
        :param user_id: Integer
        :param user_name: String
        :param dict_duration: Dictionary, {'start_date': (YYYYMMDD), 'end_date': (YYYYMMDD)}
        :param interval : String, one of ['custom', 'week', 'month', quarter', 'year']
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

        # print('for date in dict_duration:')

        # To check interval format
        try:
            interval = interval.lower()
            axis = ''

        except AttributeError as a:
            print(f'Invalid type: \'{type(interval)}\'. '
                  f'\'interval\' should be string. : WC.progress analyzer.get_trend_of_age')
            return "", ""

        if interval not in ['custom', 'week', 'month', 'quarter', 'year']:
            print(f"Invalid interval: \'{interval}\'. : WC.progress analyzer.get_trend_of_age")
            return "", ""


        # To change dict_duration values into 'datetime' type
        try:
            dict_duration['start_date'] = datetime.strptime(dict_duration['start_date'], '%Y%m%d')

        # when 'start_date' input is missing
        except KeyError:
            if interval in ['week', 'month', 'quarter', 'year']:
                axis = 'end_date'
                pass
            else:
                print('Please choose interval type. ERROR: WC.progression analyzer.get_trend_of_age')
                return "", ""

        except ValueError:
            print(f'Invalid date format: \'{list(dict_duration.values())[0]}\'. Format is \'YYYYMMDD\'')
            return '', ''

        try:
            dict_duration['end_date'] = datetime.strptime(dict_duration['end_date'], '%Y%m%d')

        # when 'end_date' input is missing
        except KeyError:
            if interval in ['week', 'month', 'quarter', 'year']:
                axis = 'start_date'
                pass
            else:
                print('Please choose interval type. ERROR: WC.progression analyzer.get_trend_of_age')
                return "", ""

        except ValueError:
            print(f'Invalid date format: \'{list(dict_duration.values())[1]}\'. Format is \'YYYYMMDD\'')
            return '', ''

        if axis == '' and interval == 'custom':
            pass

        elif axis == 'start_date':
            if interval == 'week':
                dict_duration['end_date'] = dict_duration[axis] + timedelta(days=6)
            elif interval == 'month':
                dict_duration['end_date'] = dict_duration[axis] + relativedelta(months=1)
            elif interval == 'quarter':
                dict_duration['end_date'] = dict_duration[axis] + relativedelta(months=3)
            elif interval == 'year':
                dict_duration['end_date'] = dict_duration[axis] + relativedelta(years=3)

        # When axis == 'end_date'
        elif axis == 'end_date':
            if interval == 'week':
                dict_duration['start_date'] = dict_duration[axis] - timedelta(days=6)
            elif interval == 'month':
                dict_duration['start_date'] = dict_duration[axis] - relativedelta(months=1)
            elif interval == 'quarter':
                dict_duration['start_date'] = dict_duration[axis] - relativedelta(months=3)
            elif interval == 'year':
                dict_duration['start_date'] = dict_duration[axis] - relativedelta(years=3)

        # seasonal_variation_function will use this durations.
        self.start_date = dict_duration['start_date']
        self.end_date = dict_duration['end_date']

        # To return all rows of predicted age result within passed duration
        age_rows = self._retrieve_user_trend(user_id, self.assess_type, dict_duration)

        if not age_rows:
            print('No age_rows')
            return '', ''

        # To get path of saved histogram graph
        histogram = self.compute_age_frequency(age_rows, user_id, user_name, dict_duration)

        # To get path of saved pie chart
        pie_chart = self.compute_age_ratio(age_rows, user_id, user_name, dict_duration)

        if histogram is not '' and pie_chart is not '':
            return histogram, pie_chart
        else:
            print("NO RETURN VALUE : WC.progress analyzer.get_trend_of_age")
            print("This error occurs when one or both of return value are []")
            return '', ''

    def compute_age_frequency(self, list_dict_result, user_id, user_name, dict_duration):
        """
        :param list_dict_result: List
        :param user_id: Integer
        :param dict_duration: Dictionary, {'start_date': 'YYYYMMDD', '
        :return: histogram: ndarray
        """

        # To get list of all predicted age
        list_age = []
        count = {}
        start_date = datetime.strftime(dict_duration['start_date'], '%Y-%m-%d')
        end_date = datetime.strftime(dict_duration['end_date'], '%Y-%m-%d')

        for i in list_dict_result:
            # print(i['result'])
            list_age.append(int(i['result']))
        # print(list_age)

        for i in list_age:
            if i not in list(count.keys()):
                count[i] = 1
            elif i in list(count.keys()):
                count[i] += 1

        # print(count)

        colors = random.choice(['green', 'darkgreen', 'navy', 'darkblue', 'slateblue', 'purple',
                                'olive', 'cadetblue', 'orange', 'mediumseagreen', 'c'])

        plt.hist(list_age, range=(min(list_age)-0.5, max(list_age)+0.5), histtype='bar',
                 rwidth=0.75, bins=len(set(list_age)), color=colors, zorder=10)

        plt.grid(True, color='gray', linestyle='dashed', linewidth=0.5, zorder=5)

        plt.xticks(range(min(list_age), max(list_age)+1))
        plt.yticks(range(0, max(count.values())+5, 2))
        plt.ylabel('frequency')

        plt.xlabel('Predicted Age')
        # plt.legend(loc='upper right', ncol=1)

        plt.title(f'{user_name} \'s Predicted Age Frequency\n'
                  f'\'{start_date} ~ {end_date}\'')
        # plt.show()

        # make ndarray part
        save_path = f"./saved_chart_image/{user_id}_age_histogram.png"
        plt.savefig(save_path)

        pie_chart = cv2.imread(save_path)
        print('test')
        if pie_chart.any():
            plt.close()
            return save_path

        else:
            print("ERROR: WC.progressanalyzer.compute_emotion_frequency : histogram get None")
            return []

    def compute_age_ratio(self, list_dict_result, user_id, user_name, dict_duration):
        """
        To return pie chart
        :param list_dict_result: List
        :param user_id: Integer
        :param duration : String, one of ['week', 'month', 'quarter', 'year']
        :return: pie_chart : ndarray
        """
        start_date = datetime.strftime(dict_duration['start_date'],'%Y-%m-%d')
        end_date = datetime.strftime(dict_duration['end_date'],'%Y-%m-%d')

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

        plt.pie(ratio, labels=labels, startangle=90, autopct='%.1f%%')
        plt.title(f'{user_name} \'s Predicted Age Ratio\n'
                  f'\'{start_date} ~ {end_date}\'')
        plt.xlabel(f'Average Age : \'{round(avg_age,1)}\'')

        # make nd array part
        save_path = f"./saved_chart_image/{user_id}_age_pie_chart.png"
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

    def get_trend_of_emotions(self, user_id, user_name, dict_duration, interval='custom'):
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

        # print('for date in dict_duration:')

        # To check interval format
        try:
            interval = interval.lower()
            axis = ''

        except AttributeError as a:
            print(f'Invalid type: \'{type(interval)}\'. '
                  f'\'interval\' should be string. : WC.progress analyzer.get_trend_of_emotion')
            return "", ""

        if interval not in ['custom', 'week', 'month', 'quarter', 'year']:
            print(f"Invalid interval: \'{interval}\'. : WC.progress analyzer.get_trend_of_emotion")
            return "", ""


        # To change dict_duration values into 'datetime' type
        try:
            dict_duration['start_date'] = datetime.strptime(dict_duration['start_date'], '%Y%m%d')

        # when 'start_date' input is missing
        except KeyError:
            if interval in ['week', 'month', 'quarter', 'year']:
                axis = 'end_date'
                pass
            else:
                print('Please choose interval type. ERROR: WC.progression analyzer.get_trend_of_emotion')
                return "", ""

        except ValueError:
            print(f'Invalid date format: \'{list(dict_duration.values())[0]}\'. Format is \'YYYYMMDD\'')
            return '', ''

        try:
            dict_duration['end_date'] = datetime.strptime(dict_duration['end_date'], '%Y%m%d')

        # when 'end_date' input is missing
        except KeyError:
            if interval in ['week', 'month', 'quarter', 'year']:
                axis = 'start_date'
                pass
            else:
                print('Please choose interval type. ERROR: WC.progression analyzer.get_trend_of_emotion')
                return "", ""

        except ValueError:
            print(f'Invalid date format: \'{list(dict_duration.values())[1]}\'. Format is \'YYYYMMDD\'')
            return '', ''

        if axis == '' and interval == 'custom':
            pass

        elif axis == 'start_date':
            if interval == 'week':
                dict_duration['end_date'] = dict_duration[axis] + timedelta(days=6)
            elif interval == 'month':
                dict_duration['end_date'] = dict_duration[axis] + relativedelta(months=1)
            elif interval == 'quarter':
                dict_duration['end_date'] = dict_duration[axis] + relativedelta(months=3)
            elif interval == 'year':
                dict_duration['end_date'] = dict_duration[axis] + relativedelta(years=3)

        # When axis == 'end_date'
        elif axis == 'end_date':
            if interval == 'week':
                dict_duration['start_date'] = dict_duration[axis] - timedelta(days=6)
            elif interval == 'month':
                dict_duration['start_date'] = dict_duration[axis] - relativedelta(months=1)
            elif interval == 'quarter':
                dict_duration['start_date'] = dict_duration[axis] - relativedelta(months=3)
            elif interval == 'year':
                dict_duration['start_date'] = dict_duration[axis] - relativedelta(years=3)

        self.start_date = dict_duration['start_date']
        self.end_date = dict_duration['end_date']

        # To return all rows of predicted emotion result within passed duration
        emotion_rows = self._retrieve_user_trend(user_id, self.assess_type, dict_duration)

        if not emotion_rows:
            print('No emotion_rows')
            return '', ''

        # To get path of saved histogram graph
        histogram = self.compute_emotion_frequency(emotion_rows, user_id, user_name, dict_duration)

        # To get path of saved pie chart
        pie_chart = self.compute_emotion_ratio(emotion_rows, user_id, user_name, dict_duration)

        if histogram is not '' and pie_chart is not '':
            return histogram, pie_chart
        else:
            print("NO RETURN VALUE : WC.progress analyzer.get_trend_of_emotion")
            print("This error occurs when one or both of return value are []")
            return '', ''

    def compute_emotion_frequency(self, list_dict_result, user_id, user_name, dict_duration):
        """
        To return histogram
        :param list_dict_result: List
        :param user_id: Integer
        :param duration: String, one of ['week', 'month', 'quarter', 'year']
        :return: histogram: ndarray
        """

        start_date = datetime.strftime(dict_duration['start_date'], '%Y-%m-%d')
        end_date = datetime.strftime(dict_duration['end_date'], '%Y-%m-%d')
        list_emotion = []
        count = {}

        # To get list of detected emotion
        for i in list_dict_result:
            # print(i['result'])
            list_emotion.append(int(i['result']))

        # To get max count of frequency
        for i in list_emotion:
            if i not in list(count.keys()):
                count[i] = 1
            elif i in list(count.keys()):
                count[i] += 1

        colors = random.choice(['green', 'darkgreen', 'navy', 'darkblue', 'slateblue', 'purple',
                                'olive', 'cadetblue', 'orange', 'mediumseagreen', 'c'])

        bar = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        y_pos = np.arange(len(bar))

        plt.hist(list_emotion, range=(-0.5, 6.5), histtype='bar', rwidth=0.5, bins=len(bar), color=colors, zorder=10)

        plt.grid(True, color='gray', linestyle='dashed', linewidth=0.5, zorder=5)

        plt.ylabel('frequency')
        plt.xticks(y_pos, bar)
        plt.yticks(range(0, max(count.values())+3, 1))

        # plt.show()

        plt.title(f'{user_name} \'s Emotion Frequency\n'
                  f'\'{start_date} ~ {end_date}\'')

        # make ndarray part
        save_path = f"./saved_chart_image/{user_id}_emotion_histogram.png"
        plt.savefig(save_path)

        pie_chart = cv2.imread(save_path)

        if pie_chart.any():
            plt.close()
            return save_path

        else:
            print("ERROR: WC.progressanalyzer.compute_emotion_frequency : histogram get None")
            return []

    def compute_emotion_ratio(self, list_dict_result, user_id, user_name, dict_duration):
        """
        To return pie chart
        :param list_dict_result: List
        :param user_id: Integer
        :param duration : String, one of ['week', 'month', 'quarter', 'year']
        :return: pie_chart : ndarray
        """

        start_date = datetime.strftime(dict_duration['start_date'], '%Y-%m-%d')
        end_date = datetime.strftime(dict_duration['end_date'], '%Y-%m-%d')

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

        plt.pie(ratio, labels=labels, startangle=90, autopct='%.1f%%')
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
    print(epa.get_seasonal_variation_of_emotions(2))
    print(epa.get_seasonal_variation_of_factors(2, 1))

    # print(epa.get_seasonal_variation_of_factors(294, 'week'))
    # print(epa.get_seasonal_variation_of_emotions(294, 'week'))
    # print(epa.get_seasonal_variation_of_emotions(294, 6))
    # print(epa.get_seasonal_variation_of_emotions(2, 7))
    # print(epa.get_top_emotions(3, 8))

    # print(type(apa.get_trend_of_age(user_id=2, user_name='jisoo', duration=6)))
    # apa.get_trend_of_age(user_id=2, user_name='jisoo')
    # apa.get_trend_of_age(user_id=2, user_name='jisoo',dict_duration={'start_date': '20190701'}, interval='week')
    # apa.get_trend_of_age(user_id=2, user_name='jisoo', dict_duration={'start_date': '20190720', 'end_date': '20190731'})
    # apa.get_trend_of_age(user_id=2, user_name='jisoo', dict_duration={'end_date': '20190725'})


    # epa.get_trend_of_emotions(user_id=2, user_name='jisoo', dict_duration={'start_date': '20190720', 'end_date': '20190731'})

    # epa.get_seasonal_variation_of_emotions(294, 'week')
    # a, b = epa._retrieve_user_variation(294, 'aging_diagnosis', 'week')

    # print (a)
    # print (b)


