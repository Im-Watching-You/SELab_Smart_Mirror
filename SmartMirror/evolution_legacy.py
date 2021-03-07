
"""
Date: 2019.04.15.
Programmer: MH
Description: the module is to define functions of evolution about assessment.
"""
import operator
from datetime import datetime, timedelta
import numpy as np
from collections import Counter
import math

from db_connector import UserSessionList

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


class EvolutionAction:
    def __init__(self):
        pass

    def get_avg_age(self, ages):
        result = round(np.mean(ages))
        if math.isnan(result) or result is None:
            result = []
        return result

    def get_most_emotion(self, emotions):
        if emotions != []:
            counts = Counter(emotions)
            return [x_i for x_i, count in counts.items() if count == max(counts.values())][0]
        else:
            return []

    def get_most_frequent_age(self, ages, boundary=7):
        result = []
        counts = Counter(ages)
        counts.values()
        counts = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)
        for i in range(len(counts)):
                result.append((counts[i][0], round((counts[i][1] / len(ages)), 2)*100))
        if len(result) < boundary:
            for i in range(boundary-len(result)):
                result.append(("", 0))
        return result

    def get_most_frequent_emotion(self, emotions, boundary=7):
        result = []
        counts = Counter(emotions)
        counts.values()
        counts = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)
        for i in range(len(counts)):
            if i < boundary:
                result.append((counts[i][0], round(counts[i][1]/len(emotions), 2)*100))
        r = len(result)
        if len(result) < boundary:
            for i in range(boundary-len(result)):
                result.append(("", 0))
        return result

    def compare_data(self, list_data, days):
        time_value = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        t = datetime.strptime(time_value, '%Y-%m-%d %H:%M:%S')
        list_yst_age = []
        list_yst_emotion = []
        list_tod_age = []
        list_tod_emotion = []
        for d in list_data:
            sess_time = datetime.strptime(d['session_time'], '%Y-%m-%d %H:%M:%S')
            if (t - sess_time).days == 0:
                list_tod_age.append(d['age'])
                list_tod_emotion.append(d['emotion'])
            elif (t - sess_time).days <= days:
                list_yst_age.append(d['age'])
                list_yst_emotion.append(d['emotion'])

        l1 = self.get_most_emotion(list_yst_emotion)
        l2 = self.get_most_emotion(list_tod_emotion)

        return {'com_day': days, 'com_oth_emotion': l1, 'com_oth_age': self.get_avg_age(list_yst_age),
                'com_tod_emotion': l2, 'com_tod_age': str(self.get_avg_age(list_tod_age))}

    def compute_most_value(self, list_data, days):
        time_value = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        t = datetime.strptime(time_value, '%Y-%m-%d %H:%M:%S')
        list_age = []
        list_emotion = []
        for d in list_data:
            sess_time = datetime.strptime(d['session_time'], '%Y-%m-%d %H:%M:%S')
            if (t - sess_time).days < days:
                list_age.append(d['age'])
                list_emotion.append(d['emotion'])

        most_emotion = self.get_most_emotion(list_emotion)
        most_age = self.get_avg_age(list_age)
        return {'val_day': days, 'val_emotions': most_emotion, "val_age": most_age}

    def _sort_list(self, l1, l2, boundary=None):
        result = []
        for i in l1:
            for j in l2:
                if i[0] == j[0]:
                    result.append(j)
                    break
        if boundary is not None and len(result) < boundary:
            for i in range(boundary-len(result)):
                result.append(("", 0))
        return result

    def compare_data_cand(self, list_data, days):
        time_value = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        t = datetime.strptime(time_value, '%Y-%m-%d %H:%M:%S')
        list_yst_age = []
        list_yst_emotion = []
        list_tod_age = []
        list_tod_emotion = []
        for d in list_data:
            sess_time = datetime.strptime(d['session_time'], '%Y-%m-%d %H:%M:%S')
            if (t - sess_time).days == 0:
                list_tod_age.append(d['age'])
                list_tod_emotion.append(d['emotion'])
            elif (t - sess_time).days <= days:
                list_yst_age.append(d['age'])
                list_yst_emotion.append(d['emotion'])

        e_1 = self.get_most_frequent_emotion(list_yst_emotion)
        e_2 = self.get_most_frequent_emotion(list_tod_emotion)
        e_2 = self._sort_list(e_1, e_2, boundary=len(e_1)).copy()

        a_1 = self.get_most_frequent_age(list_yst_age)
        a_2 = self.get_most_frequent_age(list_tod_age)
        a_2 = self._sort_list(a_1, a_2, boundary=len(a_1)).copy()

        return {'com_day': days, 'com_oth_emotion': e_1, 'com_oth_age': a_1,
                'com_tod_emotion': e_2, 'com_tod_age': a_2}

    def compute_most_value_cand(self, list_data, days):
        time_value = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        t = datetime.strptime(time_value, '%Y-%m-%d %H:%M:%S')
        list_age = []
        list_emotion = []
        for d in list_data:
            sess_time = datetime.strptime(d['session_time'], '%Y-%m-%d %H:%M:%S')
            if (t - sess_time).days < days:
                list_age.append(d['age'])
                list_emotion.append(d['emotion'])

        most_emotion = self.get_most_frequent_emotion(list_emotion)
        most_age = self.get_most_frequent_age(list_age)
        return {'val_day': days, 'val_emotions': most_emotion, "val_age": most_age}

    def get_period(self, list_data):

        pass

    def get_max_emotion(self, list_data, days=5):
        emotions = []
        time_value = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        t = datetime.strptime(time_value, '%Y-%m-%d %H:%M:%S')
        for d in list_data:
            sess_time = datetime.strptime(d['session_time'], '%Y-%m-%d %H:%M:%S')
            if (t - sess_time).days <= days:
                emotions.append(d['emotion'])
        counts = Counter(emotions)
        emotion = [x_i for x_i, count in counts.items() if count == max(counts.values())]
        # return {'emotion': emotion}
        return str(counts)+str(len(emotions))

    def get_min_emotion(self, list_data, days=5):
        emotions = []
        time_value = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        t = datetime.strptime(time_value, '%Y-%m-%d %H:%M:%S')
        for d in list_data:
            sess_time = datetime.strptime(d['session_time'], '%Y-%m-%d %H:%M:%S')
            if (t - sess_time).days <= days:
                emotions.append(d['emotion'])
        counts = Counter(emotions)
        emotion = [x_i for x_i, count in counts.items() if count == min(counts.values())][0]
        return {'emotion': emotion}

    def get_median_age(self, list_data, days=5):
        ages = []
        time_value = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        t = datetime.strptime(time_value, '%Y-%m-%d %H:%M:%S')
        for d in list_data:
            sess_time = datetime.strptime(d['session_time'], '%Y-%m-%d %H:%M:%S')
            if (t - sess_time).days <= days:
                ages.append(d['age'])
        return np.median(ages)

    def compare_yesterday(self, list_data):
        time_value = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        t = datetime.strptime(time_value, '%Y-%m-%d %H:%M:%S')
        list_yst_age = []
        list_yst_emotion = []
        list_tod_age = []
        list_tod_emotion = []
        for d in list_data:
            sess_time = datetime.strptime(d['session_time'], '%Y-%m-%d %H:%M:%S')
            if (t - sess_time).days == 1:
                list_yst_age.append(d['age'])
                list_yst_emotion.append(d['emotion'])
            if (t - sess_time).days == 0:
                list_tod_age.append(d['age'])
                list_tod_emotion.append(d['emotion'])
        counts = Counter(list_yst_emotion)
        l1 = [x_i for x_i, count in counts.items() if count == max(counts.values())][0]

        counts = Counter(list_tod_emotion)
        l2 = [x_i for x_i, count in counts.items() if count == max(counts.values())][0]

        emotions = str(l1) +" -> "+ str(l2)
        ages = str(round(np.mean(list_yst_age))) + " -> " + str(round(np.mean(list_tod_age)))
        return {'emotion': emotions, 'age': ages}

    def compare_whole_data(self, list_data, days):
        time_value = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        t = datetime.strptime(time_value, '%Y-%m-%d %H:%M:%S')
        list_age_data = []
        list_emotion_data = []
        for i in range(days + 1):
            list_age_data.append([])
            list_emotion_data.append([])
        for d in list_data:
            sess_time = datetime.strptime(d['session_time'], '%Y-%m-%d %H:%M:%S')
            diff = (t-sess_time).days
            if diff <= days:
                list_age_data[diff].append(d['age'])
                list_emotion_data[diff].append(d['emotion'])

        for i in range(len(list_age_data)):
            list_age_data[i] = self.get_most_frequent_age(list_age_data[i])
        for i in range(len(list_emotion_data)):
            list_emotion_data[i] = self.get_most_frequent_emotion(list_emotion_data[i])
        emt_idx = -1
        for i in range(len(list_age_data)):
            count = 0
            for d in list_emotion_data[i]:
                if len(d[0]) > 0:
                    count += 1
            if count >= 3:
                emt_idx = i
        print(">>>> emt_idx", emt_idx, list_emotion_data[emt_idx])
        for i in range(0, len(list_age_data)):
            list_age_data[i] = self._sort_list(list_age_data[emt_idx], list_age_data[i],
                                               boundary=len(list_age_data[0])).copy()
        emt_idx = -1
        for i in range(len(list_emotion_data)):
            count = 0
            for d in list_emotion_data[i]:
                if len(d[0]) > 0:
                    count += 1
            if count >= 3:
                emt_idx = i
        for i in range(0, len(list_emotion_data)):
            list_emotion_data[i] = self._sort_list(list_emotion_data[emt_idx], list_emotion_data[i],
                                                   boundary=len(list_emotion_data[0])).copy()

        return {'com_day': days, "emotions": list_emotion_data, 'age': list_age_data}

    def draw_plot_sequence(self, d1, d2):
        x_axis = []
        picked_d1 = [[], [], []] # data of Top 3 in d1
        picked_d2 = [[], [], []] # data of Top 3 in d2
        for d in range(len(d1)):
            d_date = (datetime.today()-timedelta(days=len(d1)-d-1)).strftime('%Y-%m-%d')
            ch_date = datetime.strptime(d_date, '%Y-%m-%d').strftime('%m-%d')
            x_axis.append(ch_date)
            print('\n', d1, len(d1), len(d1[d]),d)
            picked_d1[0].append(d1[d][0][1])
            picked_d1[1].append(d1[d][1][1])
            picked_d1[2].append(d1[d][2][1])

            picked_d2[0].append(d2[d][0][1])
            picked_d2[1].append(d2[d][1][1])
            picked_d2[2].append(d2[d][2][1])

        fig = plt.figure(figsize=(4, 3), dpi=140)
        fig.add_subplot(111)
        axes = plt.gca()
        axes.set_ylim([0, 100])
        emt_idx = -1
        for i in range(len(d1)):
            count = 0
            for d in d1[i]:
                if len(d[0]) > 0:
                    count += 1
            if count >= 3:
                emt_idx = i
        plt.plot(x_axis, picked_d1[0], label=str(d1[emt_idx][0][0])+" [1st]")
        plt.plot(x_axis, picked_d1[1], label=str(d1[emt_idx][1][0])+" [2nd]")
        plt.plot(x_axis, picked_d1[2], label=str(d1[emt_idx][2][0])+" [3rd]")
        plt.legend(loc='upper left', prop={'size': 12})
        plt.show()
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data_emotion = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        fig.autofmt_xdate()
        plt.savefig('d1.png')

        for i in range(len(d2)):
            count = 0
            for d in d2[i]:
                if d[0] is not '':
                    count += 1
            if count >= 3:
                emt_idx = i
        fig = plt.figure(figsize=(4, 3), dpi=140)
        fig.add_subplot(111)
        axes = plt.gca()
        axes.set_ylim([0,100])
        plt.plot(x_axis, picked_d2[0], label=str(d2[emt_idx][0][0])+" [1st]")
        plt.plot(x_axis, picked_d2[1], label=str(d2[emt_idx][1][0])+" [2nd]")
        plt.plot(x_axis, picked_d2[2], label=str(d2[emt_idx][2][0])+" [3rd]")

        plt.legend(loc='upper left', prop={'size': 12})
        print("\n")
        for d in d2:
            print(d)
        plt.show()
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data_age = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        fig.autofmt_xdate()
        plt.savefig('d2.png')
        return data_emotion, data_age

    def draw_plot(self, d1, d2, d3, d4):
        fig = plt.figure(figsize=(4, 3), dpi=140)
        fig.add_subplot(111)
        plt.plot(['5 Days', 'Today'], [d1[0][1], d2[0][1]], label=str(d1[0][0])+" (1st)")
        plt.plot(['5 Days', 'Today'], [d1[1][1], d2[1][1]], label=str(d1[1][0])+" (2nd)")
        plt.plot(['5 Days', 'Today'], [d1[2][1], d2[2][1]], label=str(d1[2][0])+" (3rd)")
        plt.legend(loc='upper left', prop={'size': 12})
        plt.show()
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data_emotion = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        fig = plt.figure(figsize=(4, 3), dpi=140)
        fig.add_subplot(111)
        plt.plot(['5 Days', 'Today'], [d3[0][1], d4[0][1]], label=str(d3[0][0])+" (1st)")
        plt.plot(['5 Days', 'Today'], [d3[1][1], d4[1][1]], label=str(d3[1][0])+" (2nd)")
        plt.plot(['5 Days', 'Today'], [d3[2][1], d4[2][1]], label=str(d3[2][0])+" (3rd)")
        plt.legend(loc='upper left', prop={'size': 12})
        plt.show()
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data_age = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data_emotion, data_age


if __name__ == '__main__':
    ev = EvolutionAction()
    us = UserSessionList()
    l = us.load_session_list_db(2)
    # print(l)
    # print(ev.compare_yesterday(l))
    print("-=-=-=-=")
    # most = ev.compute_most_value(l, 5)
    # print(most)
    # most = ev.compute_most_value_cand(l, 5)
    # most = ev.compare_data(l, 5)
    # print(most)
    most = ev.compare_whole_data(l, 20)
    print("=== END ===")
    print(most)
    aa = ev.draw_plot_sequence(most['emotions'], most['age'])
    # aa = ev.draw_plot(most['com_oth_emotion'], most['com_tod_emotion'], most['com_oth_age'], most['com_tod_age'])
    print(aa[0].shape, aa[1].shape)
