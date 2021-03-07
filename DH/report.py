"""
Date: 2019.07.04
Programmer: DH
Description: About System Manager Report Generator
"""
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


class ReportGenerator:
    """
    To get information about age, emotion, factor from data_set, and
    make a chart from data_set.
    """
    def __init__(self):
        pass

    def get_age_avg(self, data_set, from_date, to_date, u_id=None):
        '''
        To compute age average
        :param data_set: dataframe, the data of age values
        :param from_date: String, the start date: “%Y-%m-%d”
        :param to_date: String, the end date: “%Y-%m-%d”
        :param u_id:int, user_id
        :return:int, average of age
        '''

        df_date = self.__set_df_date(data_set, from_date, to_date)
        df_u_id = self.__set_df_u_id(df_date, u_id)
        return round(df_u_id['age'].mean())

    def get_age_min(self, data_set, from_date, to_date, u_id=None):
        '''
        To compute minimum of age
        '''

        df_date = self.__set_df_date(data_set, from_date, to_date)
        df_u_id = self.__set_df_u_id(df_date, u_id)
        return df_u_id['age'].min()

    def get_age_max(self, data_set, from_date, to_date, u_id=None):
        '''
        To compute maximum of age
        '''

        df_date = self.__set_df_date(data_set, from_date, to_date)
        df_u_id = self.__set_df_u_id(df_date, u_id)
        return df_u_id['age'].max()

    def get_emotion_min(self, data_set, from_date, to_date, u_id=None):
        '''
        To compute minimum of the number of emotion
        :param data_set: dataframe, the data of emotion values
        :param from_date: String, the start date: “%Y-%m-%d”
        :param to_date: String, the end date: “%Y-%m-%d”
        :param u_id: int, user_id
        :return: dict, emotion id, the number of min emotion
        '''

        df_value = self.get_rank_type(data_set, 'emotion', from_date, to_date, u_id)

        # Get last value from ranked dataframe
        emotion = df_value.at[len(df_value) - 1, 'emotion']
        count = df_value.at[len(df_value) - 1, 'number']
        return {emotion: count}

    def get_emotion_max(self, data_set, from_date, to_date, u_id=None):
        '''
        To compute maximum of the number of emotion
        :return: dict, emotion id, the number of max emotion
        '''

        df_value = self.get_rank_type(data_set, 'emotion', from_date, to_date, u_id)

        # Get first value from ranked dataframe
        emotion = df_value.at[0, 'emotion']
        count = df_value.at[0, 'number']
        return {emotion: count}

    def get_factor_data(self, data_set, from_date, to_date, u_id=None, emotion_id=None):
        '''
        To compute each factors’ maximum, minimum, mean
        :param data_set: dataframe, the dataframe of factor values
        :param from_date: String, the start date: “%Y-%m-%d”
        :param to_date: String, the end date: “%Y-%m-%d”
        :param u_id: int, user_id
        :param emotion_id: int, emotion_id
        :return: dict, each factors’ maximum, minimum, mean
        '''

        dic_factor_data = {}
        # To check emotion_id to loc data_set
        if emotion_id is None:
            df_emotion = data_set
        else:
            df_emotion = data_set.loc[data_set['emotion'] == emotion_id].reset_index()
            del df_emotion['index']

        df_date = self.__set_df_date(df_emotion, from_date, to_date)
        df_u_id = self.__set_df_u_id(df_date, u_id)

        # get factor columns from data_set
        factor_list = [x for x in data_set.columns if 'f' in x]

        # compute max, min, mean of each factor and save in dictionary
        for i in factor_list:
            tmp_data = df_u_id[i]
            factor_data = {'max': tmp_data.max(),
                           'min': tmp_data.min(),
                           'mean': round(tmp_data.mean())
                            }
            dic_factor_data[i] = factor_data
        return dic_factor_data

    def get_rank_type(self, data_set, types, from_date, to_date, u_id=None):
        '''
        To compute rank of the number of type (age, emotion, etc)
        :param data_set: dataframe, the data of type values
        :param types: String, the type of data (age, emotion, etc)
        :param from_date: String, the start date: “%Y-%m-%d”
        :param to_date: String, the end date: “%Y-%m-%d”
        :param u_id: int, user_id
        :return: DataFrame, rank of the number of type values and percentage
        '''

        df_date = self.__set_df_date(data_set, from_date, to_date)
        df_u_id = self.__set_df_u_id(df_date, u_id)

        # using dataframe value_counts function
        value_counts = df_u_id[types].value_counts(dropna=True)
        df_value_counts = pd.DataFrame(value_counts).reset_index()
        df_value_counts.columns = [types, 'number']
        df_value_counts['freq'] = df_value_counts['number'] / df_value_counts['number'].sum()
        return df_value_counts

    def make_chart(self, data_set, types, from_date, to_date, u_id=None, date_num=3, type_num=3):
        '''
        To make chart about an age, an emotion, each factor by recorded_date
        :param data_set: dataFrame, the data of type values
        :param types: String, the type of chart (age, emotion, factor, etc)
        :param from_date: String, the start date: “%Y-%m-%d”
        :param to_date: String, the end date: “%Y-%m-%d”
        :param u_id: int, user_id
        :param date_num: int, the number of dates to show
        :param type_num: int, the number of data (age, emotion, factor) to show
        :return: image, chart image using PIL
        '''

        df_u_id = self.__set_df_u_id(data_set, u_id)

        # the rank of types
        df_ranked = self.get_rank_type(df_u_id, types, from_date, to_date)

        # if the number of type value in ranked is smaller than type_num return None
        if df_ranked.empty or len(df_ranked[types]) < type_num:
            return None

        df_chart = self.__set_df_date(df_u_id, from_date, to_date)

        # rank_list = to get value from df_ranked
        # label_list = to save each label for plot
        rank_list = []
        label_list = []
        dic_emotion = {0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Sad',5:'Surprise',6:'Neutral'}
        for i in range(type_num):
            rank_list.append(df_ranked.at[i, types])
            if types == 'emotion':
                label = 'Top ' + str(i + 1) + '. ' + dic_emotion[rank_list[i]]
            else:
                label = 'Top ' + str(i + 1) + '. ' + str(rank_list[i])
            label_list.append(label)

        # df_value_counts = rank recorded_date to find the number of each date
        df_value_counts = pd.DataFrame(df_chart['recorded_date'].value_counts()).reset_index()
        df_value_counts.columns = ['date', 'number']

        if df_value_counts.empty or len(df_value_counts['date']) < date_num:
            return None

        # date_list = to get value from df_valued_counts
        date_list = []
        for i in range(date_num):
            date_list.append(df_value_counts.at[i, 'date'])
        date_list.sort()

        # To compute types by recorded_date
        rate_list = []
        for i in range(type_num):
            data = df_chart.loc[df_chart[types] == rank_list[i]]
            for j in range(date_num):
                count = len(data.loc[data['recorded_date'] == date_list[j]])
                if types == 'age':
                    size = len(df_chart.loc[df_chart['recorded_date'] == date_list[j]])
                    rate_list.append(round(count / size * 100))
                else:
                    rate_list.append(count)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(type_num):
            ax.plot(date_list, rate_list[date_num * i:date_num * (i + 1)], label=label_list[i])
        fig.legend()

        y_axis = range(0, 101, 10)
        plt.yticks(y_axis)
        if types == 'age':
            plt.ylabel("Detected Rate (%)")
        else:
            plt.ylabel("Count Number")
        plt.xlabel("Date")

        fig.canvas.draw()
        s, (width, height) = fig.canvas.print_to_buffer()

        im = Image.frombytes("RGBA", (width, height), s)
        return im

    def make_factor_chart(self, data_set, func_types, from_date, to_date, u_id=None, date_num=3):
        '''
        To draw chart about total factors’ maximum or minimum value by recorded_date
        :param data_set: dataFrame, the dataframe of factor values
        :param func_types: String, types of function: “max”, “mean”
        :param from_date: String, the start date: “%Y-%m-%d”
        :param to_date: String, the end date: “%Y-%m-%d”
        :param u_id: int, user_id
        :param date_num: int, the number of dates to show
        :return: image, chart image using PIL
        '''

        factor_list = [x for x in data_set.columns if 'f' in x]
        dic_emotion = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

        # if emotion, get id of maximum counts of emotion
        max_emotion = {}
        if 'emotion' in data_set.columns:
            max_emotion = self.get_emotion_max(data_set, from_date, to_date, u_id)
            df_emotion = data_set.loc[data_set['emotion'] == list(max_emotion.keys())[0]].reset_index()
            del df_emotion['index']
        else:
            df_emotion = data_set

        df_u_id = self.__set_df_u_id(df_emotion, u_id)
        df_chart = self.__set_df_date(df_u_id, from_date, to_date)

        df_value_counts = pd.DataFrame(df_chart['recorded_date'].value_counts()).reset_index()
        df_value_counts.columns = ['date', 'number']

        if df_value_counts.empty or len(df_value_counts['date']) < date_num:
            return None

        date_list = []
        for i in range(date_num):
            date_list.append(df_value_counts.at[i, 'date'])
        date_list.sort()

        # compute by func_types
        list_value = []
        for i in factor_list:
            for j in range(date_num):
                df_tmp = df_chart.loc[df_chart['recorded_date'] == date_list[j]]
                if func_types == 'max':
                    max_value = df_tmp[i].max()
                    list_value.append(max_value)
                elif func_types == 'mean':
                    mean_value = round(df_tmp[i].mean())
                    list_value.append(mean_value)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(len(factor_list)):
            ax.plot(date_list, list_value[date_num * i:date_num * (i + 1)], label=factor_list[i])
        fig.legend()

        y_axis = range(0, 101, 10)
        plt.yticks(y_axis)
        if func_types == 'max':
            plt.ylabel("Max Value")
        elif func_types == 'mean':
            plt.ylabel("Mean Value")
        plt.xlabel("Date")
        if 'emotion' in data_set.columns:
            plt.title(dic_emotion[list(max_emotion.keys())[0]])

        # Save in Image attribute
        fig.canvas.draw()
        s, (width, height) = fig.canvas.print_to_buffer()

        im = Image.frombytes("RGBA", (width, height), s)
        return im

    def __set_df_date(self, data_set, from_date, to_date):
        '''
        To set dataframe by date
        '''

        # time range
        from_date = from_date + " 00:00:00"
        to_date = to_date + " 23:59:59"

        if type(data_set['recorded_date'].get(0)) == str:
            df_date = data_set.loc[(from_date <= data_set['recorded_date']) &
                                   (data_set['recorded_date'] <= to_date)]

        else:
            df_date = data_set.loc[(pd.Timestamp(from_date) <= data_set['recorded_date']) &
                                   (data_set['recorded_date'] <= pd.Timestamp(to_date))]
            df_date['recorded_date'] = df_date['recorded_date'].dt.strftime('%Y-%m-%d')

        return df_date

    def __set_df_u_id(self, data_set, u_id):
        '''
        To set dataframe by user id
        '''
        if u_id is None:
            df_u_id = data_set
        else:
            df_u_id = data_set.loc[data_set['u_id'] == u_id].reset_index()
            del df_u_id['index']

        return df_u_id
