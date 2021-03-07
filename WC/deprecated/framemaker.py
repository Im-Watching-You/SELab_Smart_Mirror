# To get data frame from Data Base
# author  : Woo Chan Park
# version : 2019.06.19

import WC.dbconnector as db
import pandas as pd
from IPython.display import display


# Super Class
class FrameMaker:

    # Constructor
    # To make a instance means to refresh it's data frame : defined as class attribute
    def __init__(self):
        self.conn = None
        db.connect_to_db(self)
        self.refresh_entire_table()
        db.disconnect_from_db(self)
        del self

    def refresh_table(self):
        pass


# Sub Class 1
class UsersFrame(FrameMaker):

    # class attribute
    df = pd.DataFrame(columns=['user_id', 'password', 'gender', 'age', 'birth_date', 'first_name', 'last_name', 'phone_number', 'email', 'joined_date'])

    # When this function returns True, it means that class attribute 'UserFrame.df' has been refreshed completely
    def refresh_entire_table(self, order_by=None):
        with self.conn.cursor() as cursor:

            if order_by is None:
                sql = 'SELECT * FROM users'
            elif order_by is 'Rand':
                sql = 'SELECT * FROM users ORDER BY rand()'
            elif order_by is 'male_first':
                sql = 'SELECT * FROM users ORDER BY gender ASC'
            elif order_by is 'female_first':
                sql = 'SELECT * FROM users ORDER BY gender DESC'
            elif order_by is 'Birth_asc':
                sql = 'SELECT * FROM users ORDER BY birth_date ASC'     # older first
            elif order_by is 'Birth_desc':
                sql = 'SELECT * FROM users ORDER BY birth_date DESC'    # younger first
            elif order_by is 'joined_date':
                sql = 'SELECT * FROM users ORDER BY joined_date ASC'
            elif order_by is 'joined_date':
                sql = 'SELECT * FROM users ORDER BY joined_date DESC'
            else:
                print("Invalid ordering option")
                return False

            try:
                cursor.execute(sql)
                result = cursor.fetchall()
                UsersFrame.df = pd.DataFrame(result, columns=['user_id', 'password', 'gender', 'age', 'birth_date', 'first_name', 'last_name', 'phone_number', 'email', 'joined_date'])
                return True

            except Exception as e:
                return False

    # type of result : 'list'
    # fetchall() returns 'list and fetchone() returns 'dict
    # method below do works with 'list' type
    @classmethod
    def make_users_frame(cls, result):
        return pd.DataFrame(result, columns=['user_id', 'password', 'gender', 'age', 'birth_date', 'first_name', 'last_name', 'phone_number', 'email', 'joined_date'])

# Sub Class 2
class AgeFrame(FrameMaker):

    # class attribute
    df = pd.DataFrame(columns=['da_id', 'user_id', 'age', 'saved_path', 'recorded_date'])

    # When this function returns True, it means that class attribute 'AgeFrame.df' has been refreshed completely
    def refresh_entire_table(self, order_by=None):
        with self.conn.cursor() as cursor:

            if order_by is None:
                sql = 'SELECT * FROM detected_age'
            elif order_by is 'Rand':
                sql = 'SELECT * FROM detected_age ORDER BY rand()'
            elif order_by is 'Age_asc':
                sql = 'SELECT * FROM detected_age ORDER BY age ASC'     # older first
            elif order_by is 'Age_desc':
                sql = 'SELECT * FROM detected_age ORDER BY age DESC'    # younger first
            elif order_by is 'User_id_asc':
                sql = 'SELECT * FROM detected_age ORDER BY user_id ASC'
            elif order_by is 'User_id_desc':
                sql = 'SELECT * FROM detected_age ORDER BY user_id DESC'

            else:
                print("Invalid ordering option")
                return False

            try:
                cursor.execute(sql)
                result = cursor.fetchall()
                AgeFrame.df = pd.DataFrame(result, columns=['da_id', 'user_id', 'age', 'saved_path', 'recorded_date'])
                return True

            except Exception as e:
                return False

    # type of result : 'list'
    @classmethod
    def make_age_frame(cls, result):

        return pd.DataFrame(result, columns=['da_id', 'user_id', 'age', 'saved_path', 'recorded_date'])


# Sub Class 3
class EmotionFrame(FrameMaker):

    # class attribute
    df = pd.DataFrame(columns=['de_id', 'user_id', 'emotion', 'saved_path', 'recorded_date'])

    # When this function returns True, it means that class attribute 'EmotionFrame.df' has been refreshed completely
    def refresh_entire_table(self, order_by):
        with self.conn.cursor() as cursor:

            if order_by is None:
                sql = 'SELECT * FROM detected_emotion'
            elif order_by is 'Rand':
                sql = 'SELECT * FROM detected_emotion ORDER BY rand()'

            try:
                cursor.execute(sql)
                result = cursor.fetchall()
                EmotionFrame.df = pd.DataFrame(result, columns=['de_id', 'user_id', 'emotion', 'saved_path', 'recorded_date'])
                return True

            except Exception as e:
                return False

    # type of result : 'list'
    @classmethod
    def make_emotion_frame(cls, result):

        return pd.DataFrame(result, columns=['de_id', 'user_id', 'emotion', 'saved_path', 'recorded_date'])


# Sub Class 4
class FeedbackFrame(FrameMaker):

    # class attribute
    df = pd.DataFrame(columns=['fb_id', 'user_id', 'rm_id', 'feedback_rating', 'rated_date'])

    # When this function returns True, it means that class attribute 'FeedbackFrame.df' has been refreshed completely
    def refresh_entire_table(self):
        with self.conn.cursor() as cursor:
            sql = 'SELECT * FROM feedback_result'
            try:
                cursor.execute(sql)
                result = cursor.fetchall()
                FeedbackFrame.df = pd.DataFrame(result, columns=['fb_id', 'user_id', 'rm_id', 'feedback_rating', 'rated_date'])
                return True

            except Exception as e:
                return False

    # type of result : 'list'
    @classmethod
    def make_feedback_frame(cls, result):

        return pd.DataFrame(result, columns=['fb_id', 'user_id', 'rm_id', 'feedback_rating', 'rated_date'])


# Sub Class 5
class RemedyFrame(FrameMaker):

    # class attribute
    df = pd.DataFrame(columns=['rm_id', 'rm_type', 'symptom', 'provider', 'url', 'maintain', 'improve', 'prevent',
                               'description', 'edit_date'])

    # Returns Boolean
    def refresh_entire_table(self):
        with self.conn.cursor() as cursor:
            sql = 'SELECT * FROM remedy_method'
            try:
                cursor.execute(sql)
                result = cursor.fetchall()
                RemedyMethodFrame.df = pd.DataFrame(result, columns=['rm_id', 'rm_type', 'symptom', 'provider', 'url', 'maintain', 'improve', 'prevent', 'description', 'edit_date'])
                return True

            except Exception as e:
                return False

    # type of result : 'list'
    @classmethod
    def make_remedy_frame(cls, result):

        return pd.DataFrame(result, columns=['rm_id', 'rm_type', 'symptom', 'provider', 'url', 'maintain', 'improve', 'prevent', 'description', 'edit_date'])


# df = UsersFrame()
# display(UsersFrame.df)

# df = RemedyMethodFrame()
# display(RemedyMethodFrame.df)
