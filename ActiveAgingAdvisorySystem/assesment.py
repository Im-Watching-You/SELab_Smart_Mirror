import WC.dbconnector as db
import WC.stringmaker as sm

import random
from datetime import datetime
from datetime import timedelta

import pymysql


class Assessment:

    def __init__(self):
        self.conn = None
        db.connect_to_db(self)

    def __del__(self):
        db.disconnect_from_db(self)

    def _db_register_assessment(self, assess_type, dict_info):
        """
        :param assess_type:
            String, a type of assessment in ("aging_diagnosis", "emotion_diagnosis", "aging_recommendation")
        :param dict_info: Dictionary
            key(1): model_id: Integer
            key(2): photo_id: Integer
            key(3): session_id: Integer
            key(4): result: String
        :return: assess_id: Integer, id of registered row
        """

        for i in list(dict_info):
            if i is 'model_id':
                m_id = dict_info[i]

            elif i is 'photo_id':
                p_id = dict_info[i]

            elif i is 'session_id':
                s_id = dict_info[i]

            elif i is 'result':
                result = dict_info[i]

        time = datetime.now()

        sql = f'INSERT INTO assessment (assess_type, m_id, p_id, s_id, result, recorded_date) ' \
              f'VALUES (\'{assess_type}\', {m_id}, {p_id}, {s_id}, \'{result}\', \'{time}\')'

        sql_get_id = f"SELECT id FROM assessment " \
                     f"WHERE m_id = {m_id} AND p_id ={p_id} AND s_id={s_id} AND result = {result} " \
                     f"AND assess_type = \'{assess_type}\' " \
                     f"ORDER BY recorded_date DESC limit 1"

        with self.conn.cursor() as cursor:
            try:
                cursor.execute(sql)
                # print("Assessment Data has been Inserted.")

                cursor.execute(sql_get_id)
                assess_id = cursor.fetchall()[0]['id']
                # print(assess_id)
                return assess_id

            except pymysql.err.Error as i:
                # print(i)
                print("Error: _db_register_assessment")
                return -1

    def _db_deregister_assessment(self, assess_type, id):
        """
        To delete a row in the table
        :param assess_type:
            String, a type of assessment in ("aging_diagnosis", "emotion_diagnosis", "aging_recommendation")
        :param id: Integer, id of assessment table
        :return: Boolean
        """

        sql = f'DELETE FROM assessment WHERE id = {id} AND assess_type = \'{assess_type}\''
        sql_count_rows = f"Select count(id) from smart_mirror_system.assessment"

        with self.conn.cursor() as cursor:
            try:
                cursor.execute(sql_count_rows)
                pre_num_of_rows = cursor.fetchall()[0]['count(id)']
                # print(pre_num_of_rows)

                cursor.execute(sql)

                cursor.execute(sql_count_rows)
                post_num_of_rows = cursor.fetchall()[0]['count(id)']
                # print(post_num_of_rows)

                if post_num_of_rows == pre_num_of_rows:
                    print(f"There is no matching ID with assess_type: \'{assess_type}\'")

                else:
                    print(f"The model has been deleted.")

                return True

            except pymysql.err.IntegrityError as i:
                print(i)
                print("ERROR: _db_deregister_assessment")
                return False

    def _db_retrieve_assessments(self, assess_type, dict_condition=None):
        """
        :param assess_type:
            String, a type of assessment in ("aging_diagnosis", "emotion_diagnosis", "aging_recommendation")
        :param dict_condition:
            key(1) start_date: String with "YYYYMMDD" format
            key(2) end_date: String with "YYYYMMDD" format
        :return:
        """
        start_date = '20000101'
        end_date = '20991231'

        if dict_condition is None:
            dict_condition = dict.fromkeys(['start_date', 'end_date'])
            dict_condition.update({'start_date': '20000101', 'end_date': '20991231'})

        for i in list(dict_condition):
            if i is 'start_date':
                start_date = dict_condition[i]
            elif i is 'end_date':
                end_date = dict_condition[i]

        # To include end_date : because 2019-07-01 means 2019-07-01 00:00:00.
        end_date = datetime.strptime(end_date, "%Y%m%d") + timedelta(days=1)

        duration = f"recorded_date >= \'{start_date}\' AND recorded_date < \'{end_date}\'"

        sql = f'SELECT * FROM smart_mirror_system.assessment WHERE assess_type = \'{assess_type}\' AND '

        with self.conn.cursor() as cursor:
            try:
                # print(sql+duration)
                cursor.execute(sql+duration)
                result = cursor.fetchall()

                if result is ():
                    return []
                else:
                    return result

            except Exception as e:
                print(e)
                print('ERROR : _db_retrieve_assessments')
                return False

    def _db_retrieve_assessment_by_ids(self, assess_type, dict_condition):
        """

        :param assess_type:
        :param dict_condition:
            key(1) id: Integer
            key(2) model_id: Integer
            key(3) session_id: Integer
        :return:
        """

        id, m_id, s_id = None, None, None
        if dict_condition is not None:
            for i in list(dict_condition):
                if i is 'id':
                    id = dict_condition[i]
                elif i is 'model_id':
                    m_id = dict_condition[i]
                elif i is 'session_id':
                    s_id = dict_condition[i]
                else:
                    print("Please check your key name again.")
                    print(f"\'{i}\' is invalid.")
                    return []

        sql = f'SELECT * FROM smart_mirror_system.assessment WHERE assess_type = \'{assess_type}\' '

        if id is None and m_id is None and s_id is None:
            condition = ""

        elif id is not None and m_id is None and s_id is None:
            condition = f'AND id = \'{id}\''
        elif id is not None and m_id is not None and s_id is None:
            condition = f'AND id = \'{id}\' AND m_id = \'{m_id}\''
        elif id is not None and m_id is None and s_id is not None:
            condition = f'AND id = \'{id}\' AND s_id = \'{s_id}\''
        elif id is not None and m_id is not None and s_id is not None:
            condition = f'AND id = \'{id}\' AND m_id = \'{m_id}\' And s_id = \'{s_id}\''

        elif m_id is not None and s_id is None:
            condition = f'AND m_id = \'{m_id}\''

        elif m_id is not None and s_id is not None:
            condition = f'AND m_id = \'{m_id}\' AND s_id = {s_id}'

        elif m_id is None and s_id is not None:
            condition = f'AND s_id = \'{s_id}\''

        with self.conn.cursor() as cursor:
            try:
                cursor.execute(sql+condition)
                # print(sql+condition)
                result = cursor.fetchall()

                if result is ():
                    return []
                else:
                    return result

            except Exception as e:
                print(e)
                return []

    def _db_retrieve_latest_assessment_by_ids(self, assess_type, dict_condition):
        """
        To get latest assessment that have been made matching with given ids.
        :param dict_condition:
            key(1): m_id : Integer
            key(2): p_id : Integer
            key(3): s_id : Integer
        :return:
        """
        m_id = dict_condition['model_id']
        p_id = dict_condition['photo_id']
        s_id = dict_condition['session_id']

        if m_id and p_id and s_id:
            sql = f'SELECT id FROM smart_mirror_system.assessment ' \
                  f'WHERE m_id = \'{m_id}\' AND p_id = \'{p_id}\' AND s_id = \'{s_id}\' ' \
                  f'AND assess_type = \'{assess_type}\''
        else:
            print("Parameters are missing. key(1): m_id : Integer, key(2): p_id : Integer, key(3): s_id : Integer ")
            return []

        with self.conn.cursor() as cursor:
            try:
                # print(sql)
                cursor.execute(sql)
                result = cursor.fetchall()

                if result is ():
                    return []
                else:
                    return result[0]['id']

            except Exception as e:
                print(e)
                return []

    def _db_update_assessment(self, assess_type, id, dict_condition):

        sql = f"UPDATE assessment SET {sm.StringMaker.key_equation_value(dict_condition)} " \
              f"WHERE id = {id} AND assess_type = \'{assess_type}\'"

        with self.conn.cursor() as cursor:
            try:
                is_updated = cursor.execute(sql)
                if is_updated:
                    print("Assessment Data has been Updated.")
                else:
                    print(f"There is no matching id or Input data has already been applied.")
                return True

            except pymysql.err.IntegrityError as i:
                print(i)
                return False

    def _db_make_dummy_assessment(self, assess_type, count):

        with self.conn.cursor() as cursor:

            for _ in range(count):
                if assess_type is 'aging_diagnosis':
                    result = random.randrange(20, 40)

                elif assess_type is 'emotion_diagnosis':
                    result = random.randrange(0, 6)
                    # 0: Angry, 1: Disgust, 2: Fear, 3: Happy, 4: Sad, 5: Surprise, 6: Natural

                elif assess_type is 'aging_recommendation':
                    result = random.randrange(13, 132)

                sql = f'INSERT INTO assessment (assess_type, m_id, p_id, s_id, result, recorded_date) ' \
                      f'VALUES (\'{assess_type}\', {random.randrange(100,140)}, {random.randrange(1,108)}, ' \
                      f'{random.randrange(1,300)}, \'{result}\', \'{datetime.now()}\')'

                try:
                    cursor.execute(sql)
                    print("Assessment Data has been Inserted.")

                except pymysql.err.IntegrityError as i:
                    print(i)
                    print("Error: _db_register_assessment")
                    return False

            return True

    def _db_truncate_assessment(self, assess_type):

        option = input(f'<STAFF ONLY> This method will delete entire rows of \'{assess_type}\' data. Are you Sure? (Y/N)')

        if option.lower() == 'y':
            with self.conn.cursor() as cursor:
                try:
                    sql = f"DELETE FROM smart_mirror_system.assessment WHERE assess_type = \'{assess_type}\'"
                    cursor.execute(sql)
                    print(f"All \'{assess_type}\' type data has been truncated.")
                    return True

                except Exception as e:
                    print(e)
                    return False
        else:
            print('Truncate Canceled.')
            return False


class AgingDiagnosis(Assessment):

    assess_type = 'aging_diagnosis'

    def register_assessment(self, dict_info):
        """
        :param dict_info: Dictionary
            key(1): model_id: Integer
            key(2): photo_id: Integer
            key(3): session_id: Integer
            key(4): result: String
        :return: Integer, id or -1
        """
        return self._db_register_assessment(self.assess_type, dict_info)

    def deregister_assessment(self, id):
        """

        :param id: Integer
        :return: boolean
        """
        return self._db_deregister_assessment(self.assess_type, id)

    def retrieve_assessments(self, dict_condition=None):
        """
        :param dict_condition:
            key(1) start_date: String with "YYYYMMDD" format
            key(2) end_date: String with "YYYYMMDD" format
        :return:
        """

        return self._db_retrieve_assessments(self.assess_type, dict_condition)

    def retrieve_assessment_by_ids(self, dict_condition=None):
        """

        :param dict_condition:
            key(1) id: Integer
            key(2) model_id: Integer
            key(3) session_id: Integer
        :return: list of dictionaries
        """
        return self._db_retrieve_assessment_by_ids(self.assess_type, dict_condition)

    def retrieve_latest_assessment_by_ids(self, dict_condition):
        """
        To get latest assessment that have been made matching with given ids.
        :param dict_condition:
            key(1): m_id : Integer
            key(2): p_id : Integer
            key(3): s_id : Integer

        :Test Code
            ad.retrieve_latest_assessment_by_ids({"model_id": 138,"photo_id":21,"session_id":292})
        :return: id : Integer
        """
        return self._db_retrieve_latest_assessment_by_ids(self.assess_type, dict_condition)

    # Special method
    def retrieve_age_by_user_id(self, user_id):
        return self._db_retrieve_age_by_user_id(user_id)

    def update_assessment(self, id, dict_condition):
        """
        To change predicted recommendation for training model
        :param id: id of assessment
        :param dict_condition: any attributes that want to change
        :return:
        """
        return self._db_update_assessment(self.assess_type, id, dict_condition)

    def make_dummy_assessment(self, count=10):
        return self._db_make_dummy_assessment(self.assess_type, count)

    def truncate_assessment(self):
        return self._db_truncate_assessment(assess_type=self.assess_type)

    def _db_retrieve_age_by_user_id(self, user_id):

        user = "smart_mirror_system.user"
        session = "smart_mirror_system.session"
        photo = "smart_mirror_system.photo"
        assessment = "smart_mirror_system.assessment"
        with self.conn.cursor() as cursor:
            try:
                sql = f"SELECT user.age " \
                    f"FROM {user} join {session} join {photo} join {assessment} " \
                    f"WHERE {user}.id = {session}.u_id AND {session}.id = {photo}.s_id AND " \
                    f"{photo}.id = {assessment}.p_id and {photo}.s_id = {assessment}.s_id and {user}.id = {user_id}"

                # print(sql)
                cursor.execute(sql)
                result = cursor.fetchall()

                if result is ():
                    return []
                else:
                    return result[0]['age']

            except Exception as e:
                print(e)
                return []


class EmotionDiagnosis(Assessment):

    assess_type = 'emotion_diagnosis'

    def register_assessment(self, dict_info):
        """
        :param dict_info: Dictionary
            key(1): model_id: Integer
            key(2): photo_id: Integer
            key(3): session_id: Integer
            key(4): result: String
        :return: Integer, id or -1
        """
        return self._db_register_assessment(self.assess_type, dict_info)

    def deregister_assessment(self, id):
        """

        :param id: Integer
        :return: boolean
        """
        return self.deregister_assessment(self.assess_type, id)

    def retrieve_assessments(self, dict_condition=None):
        """
        :param dict_condition:
            key(1) start_date: String with "YYYYMMDD" format
            key(2) end_date: String with "YYYYMMDD" format
        :return:
        """
        return self._db_retrieve_assessments(self.assess_type, dict_condition)

    def retrieve_assessment_by_ids(self, dict_condition=None):
        """

        :param dict_condition:
            key(1) id: Integer
            key(2) model_id: Integer
            key(3) session_id: Integer
        :return: list of dictionaries
        """
        return self._db_retrieve_assessment_by_ids(self.assess_type, dict_condition)

    def retrieve_latest_assessment_by_ids(self, dict_condition):
        """
        To get latest assessment that have been made matching with given ids.
        :param dict_condition:
            key(1): m_id : Integer
            key(2): p_id : Integer
            key(3): s_id : Integer

        :Test Code
             ed.retrieve_latest_assessment_by_ids({"model_id": 138,"photo_id":83,"session_id":53})
        :return: id : Integer
        """
        return self._db_retrieve_latest_assessment_by_ids(self.assess_type, dict_condition)

    def update_assessment(self, id, dict_condition):
        """
        To change predicted recommendation for training model
        :param id: id of assessment
        :param dict_condition: any attributes that want to change
        :return:
        """
        return self._db_update_assessment(self.assess_type, id, dict_condition)

    def make_dummy_assessment(self, count=10):
        return self._db_make_dummy_assessment(self.assess_type, count)

    def truncate_assessment(self):
        return self._db_truncate_assessment(assess_type=self.assess_type)


class AgingRecommendation(Assessment):

    assess_type = 'aging_recommendation'

    def register_assessment(self, dict_info):
        """
        :param dict_info: Dictionary
            key(1): model_id: Integer
            key(2): photo_id: Integer
            key(3): session_id: Integer
            key(4): result: String
        :return: boolean
        """
        return self._db_register_assessment(self.assess_type, dict_info)

    def deregister_assessment(self, id):
        """

        :param id: Integer
        :return: boolean
        """
        return self._db_deregister_assessment(self.assess_type, id)

    def retrieve_assessments(self, dict_condition=None):
        """

        :param dict_condition:
        :return:
        """
        return self._db_retrieve_assessments(self.assess_type, dict_condition)

    def retrieve_assessment_by_ids(self, dict_condition=None):
        """

        :param dict_condition:
            key(1) id: Integer
            key(2) model_id: Integer
            key(3) session_id: Integer
        :return: list of dictionaries
        """
        return self._db_retrieve_assessment_by_ids(self.assess_type, dict_condition)

    def update_assessment(self, id, dict_condition):
        """
        To change predicted recommendation for training model
        :param id: id of assessment
        :param dict_condition: any attributes that want to change
        :return:
        """
        return self._db_update_assessment(self.assess_type, id, dict_condition)

    def make_dummy_assessment(self, count=10):
        return self._db_make_dummy_assessment(self.assess_type, count)

    def truncate_assessment(self):
        return self._db_truncate_assessment(assess_type=self.assess_type)


if __name__ == '__main__':
    ad = AgingDiagnosis()
    ed = EmotionDiagnosis()
    ar = AgingRecommendation()

    # print(ad.register_assessment({"model_id": 1, "photo_id": '980983', "session_id": 30, "result": 21}))
    print(ed.register_assessment({"model_id": 1, "photo_id": 23, "session_id": 30, "result": 20}))
    # ar.register_assessment({"model_id": 1, "photo_id": 23, "session_id": 30, "result": 20})

    # ad.deregister_assessment(17)
    # ed.deregister_assessment(id=4)
    # ar.deregister_assessment(id=4)

    # print(len(ad.retrieve_assessments({"start_date": '20190709', 'end_date': '20190709'})))
    # print(len(ad.retrieve_assessments({"start_date": '20190709', 'end_date': '20190710'})))

    # ad.retrieve_latest_assessment_by_ids({"model_id": 138,"photo_id":21,"session_id":292})
    # ed.retrieve_latest_assessment_by_ids({"model_id": 138,"photo_id":83,"session_id":53})

    # print(ad.retrieve_age_by_user_id(95))
    # ad.retrieve_assessment_by_ids({'id': 42})
    # ed.retrieve_assessment_by_ids({'id': 42})
    # ar.retrieve_assessment_by_ids({'id': 42})

    # ad.update_assessment(19, {"m_id": '31'})  # Code to input same data already in db
    # ad.update_assessment(17, {"m_id": '31'})    # Code to check valid input format but non existing id
    # ed.update_assessment(1, {"result": '30'})
    # ar.update_assessment(1, {"result": '30'})

    # ad.make_dummy()
    # ed.make_dummy()
    # ar.make_dummy()

    # ad.truncate_assessment()
    # ed.truncate_assessment()
    # ar.truncate_assessment()
