import WC.dbconnector as db
import WC.stringmaker as sm
import WC.dummymaker as dm
from datetime import datetime
from dateutil.relativedelta import relativedelta
import random
from datetime import timedelta


class MlModel:

    def __init__(self):
        self.conn = None
        db.connect_to_db(self)

    def __del__(self):
        db.disconnect_from_db(self)

    def _db_register_model(self, model_type, dict_info):
        """
        :param model_type:
            String, one of ['aging_model', 'emotion_model', 'recommendation_model', 'face_recognition_model']
        :param dict_info:
            key(1): training_set_id: Integer
            key(2): staff_id: Integer
            key(3): num_of_data: Integer
            key(4_: saved_path: String
        :return: boolean
        """

        for i in list(dict_info):
            if i is 'training_set_id':
                t_id = dict_info[i]

            elif i is 'staff_id':
                s_id = dict_info[i]

            elif i is 'num_of_data':
                num_of_data = dict_info[i]

            elif i is 'saved_path':
                saved_path = dict_info[i]

        sql = f"INSERT INTO smart_mirror_system.ml_model (model_type, t_id, s_id, num_of_data, released_date, " \
              f"last_updated_date, saved_path) " \
              f"VALUES (\'{model_type}\', {t_id}, {s_id}, {num_of_data}, " \
              f"\'{datetime.now()}\', \'{datetime.now()}\', " \
              f"\'{saved_path}\')"
        # print(sql)

        with self.conn.cursor() as cursor:
            try:
                cursor.execute(sql)
                print(f"\'{model_type}\' has been registered.")
                return True

            except Exception as e:
                print(e)
                return False

    def _db_deregister_model(self, model_type, model_id):
        """

        :param model_type:
            String, one of ['aging_model', 'emotion_model', 'recommendation_model', 'face_recognition_model']
        :param model_id: Integer
        :return:
        """

        sql = f"DELETE FROM smart_mirror_system.ml_model WHERE id = {model_id} and model_type = \'{model_type}\'"
        sql_count_rows = f"Select count(id) from smart_mirror_system.ml_model"

        with self.conn.cursor() as cursor:
            try:
                # print(sql)
                # print(sql_count_rows)

                cursor.execute(sql_count_rows)
                pre_num_of_rows = cursor.fetchall()[0]['count(id)']
                # print(pre_num_of_rows)
                cursor.execute(sql)

                cursor.execute(sql_count_rows)
                post_num_of_rows = cursor.fetchall()[0]['count(id)']
                # print(post_num_of_rows)

                if post_num_of_rows == pre_num_of_rows:
                    print(f"There is no matching ID with model_type: {model_type}")

                else:
                    print(f"The model has been deleted.")

                return True

            except Exception as e:
                print(e)
                return False

    def _db_retrieve_models(self, model_type, staff_id=None, condition=""):
        """

        :param staff_id: Integer
        :param condition: String : partial sql statement
        :return:
        """
        # print(condition)
        with self.conn.cursor() as cursor:
            try:
                if staff_id is not None and model_type is not 'recommendation_model':
                    sql = f'SELECT * FROM smart_mirror_system.ml_model WHERE s_id = {staff_id} AND model_type = \'{model_type}\' AND '

                elif model_type is 'recommendation_model' and staff_id is None:
                    sql = f'SELECT * FROM smart_mirror_system.ml_model WHERE '

                elif model_type is 'recommendation_model' and staff_id is not None:
                    sql = f'SELECT * FROM smart_mirror_system.ml_model where s_id = \'{staff_id}\' AND'

                else:
                    sql = f'SELECT * FROM smart_mirror_system.ml_model where model_type = \'{model_type}\' AND'

                # print('testing sql (line 86) :\n' + sql + str(condition))
                # print(sql+str(condition))
                cursor.execute(sql + str(condition))
                result = cursor.fetchall()

                if result is ():
                    return []

                else:
                    return result

            except Exception as e:
                print(e)
                print("_db_retrieve_models: error")
                return []

    def _db_retrieve_model_by_ids(self, id=None, training_set_id=None, staff_id=None):
        """

        :param model_type:
            String, one of ['aging_model', 'emotion_model', 'recommendation_model', 'face_recognition_model']
        :param id: Integer
        :param training_set_id: Integer
        :param staff_id: Integer
        :return: List of Dictionaries
        """

        sql = f"SELECT * FROM smart_mirror_system.ml_model "

        if id is None and training_set_id is None and staff_id is None:
            condition = ""
        elif id is not None and training_set_id is None and staff_id is None:
            condition = f"WHERE id = {id}"
        elif id is not None and training_set_id is not None and staff_id is None:
            condition = f"WHERE id = {id} AND t_id = {training_set_id}"
        elif id is not None and training_set_id is None and staff_id is not None:
            condition = f"WHERE id = {id} AND s_id = {staff_id}"
        elif id is not None and training_set_id is not None and staff_id is not None:
            condition = f"WHERE id = {id} AND t_id = {training_set_id} AND s_id = {staff_id}"

        elif training_set_id is not None and staff_id is not None:
            condition = f"WHERE t_id = {training_set_id} AND s_id = {staff_id}"
        elif training_set_id is None and staff_id is not None:
            condition = f"WHERE s_id = {staff_id}"
            # print(sql+condition)
        elif training_set_id is not None and staff_id is None:
            condition = f"WHERE t_id = {training_set_id}"

        with self.conn.cursor() as cursor:
            try:
                # print(sql+condition)
                cursor.execute(sql + condition)
                result = cursor.fetchall()

                if result is ():
                    return []
                else:
                    return result

            except Exception as e:
                print(e)
                print('_db_retrieve_by_ids : error')
                return []

    def _db_retrieve_latest_model_id(self, model_type):
        """
        To get Latest version of model id
        :param model_type:
            String, one of ['aging_model', 'emotion_model', 'recommendation_model', 'face_recognition_model']
        :return: [{"id" : #}]
        """

        sql = f"SELECT id FROM smart_mirror_system.ml_model WHERE model_type = \'{model_type}\' " \
            f"ORDER BY last_updated_date DESC limit 1"

        with self.conn.cursor() as cursor:
            try:
                cursor.execute(sql)
                result = cursor.fetchall()

                if result is ():
                    return []
                else:
                    return result[0]['id']

            except Exception as e:
                print(e)
                print("ERROR : __db_retrieve_latest_model_id")
                return False

    def _db_retrieve_model_avg_rating(self, model_type, model_id):
        """

        :param model_type:
            String, one of ['aging_model', 'emotion_model', 'recommendation_model', 'face_recognition_model']
        :param model_id:
        :return: list of Dictionaries
        """
        model_type = model_type.split('_')[0]

        sql =f"SELECT model_id, avg(rating) FROM SMART_MIRROR_SYSTEM.{model_type}_maf " \
            f"WHERE model_id = {model_id} GROUP BY model_id"

        with self.conn.cursor() as cursor:
            try:
                cursor.execute(sql)
                result = cursor.fetchall()

                if result is ():
                    return []
                else:
                    return result

            except Exception as e:
                print(e)
                print("_db_retrieve_model_rating : error")

                return []

    def _db_update_model(self, model_id, dict_info):
        """

        :param model_type:
            String, one of ['aging_model', 'emotion_model', 'recommendation_model', 'face_recognition_model']
        :param model_id: Integer
        :param dict_info: Any attributes that you want to change
        :return:
        """
        cursor = self.conn.cursor()

        # To check dict_info has format problem
        condition_string = sm.StringMaker().key_equation_value(dict_info)
        if condition_string is "":
            print("Invalid input in dict_info. Value of item should be String type.")
            return False

        # To validate dict_info's keys
        for i in dict_info:
            if i not in ('model_type', 't_id', 's_id', 'num_of_data', 'accuracy', 'last_updated_date', 'saved_path'):
                print(f"Invalid key type \'{i}\'. Please check your key again.")
                return False

        sql = f"UPDATE ml_model SET {sm.StringMaker().key_equation_value(dict_info)} " \
              f"WHERE id = {model_id} "

        try:
            is_updated = cursor.execute(sql)
            if is_updated:
                print(f"The model with id : \'{model_id}\' has been updated.")
            else:
                print(f"There is no matching id or Input data has already been applied.")
            return True

        except Exception as e:
            print(e)
            return False

    def _db_make_dummy_model(self, model_type, count):

        for _ in range(count):
            random_date = datetime.strptime(dm.dummy_date(start_year=2018), "%Y%m%d%H")
            sql = f"INSERT INTO smart_mirror_system.ml_model (model_type, t_id, s_id, num_of_data," \
                  f" accuracy, released_date, last_updated_date, saved_path) " \
                  f"VALUES (\'{model_type}\', {random.randrange(1,100)}, {random.randrange(1,5)}, " \
                  f"{random.randrange(100,10000)}, {random.randrange(70,99)}, \'{random_date}\', " \
                  f"\'{random_date + timedelta(days=random.randrange(1,30)) + timedelta(hours=random.randrange(1,24)) + timedelta(minutes=random.randrange(1,60))}\', "\
                  f"\'{'/root'+str(random.randrange(1000,9999))}\')"

            # print(sql)
            with self.conn.cursor() as cursor:
                try:
                    cursor.execute(sql)

                except Exception as e:
                    print(e)
                    return False

        return True

    def _db_truncate_model(self, model_type):

        option = input(f'<STAFF ONLY> This method will delete entire rows of \'{model_type}\' data. Are you Sure? (Y/N)')

        if option.lower() == 'y':
            with self.conn.cursor() as cursor:
                try:
                    sql = f"Delete FROM smart_mirror_system.ml_model WHERE model_type = \'{model_type}\'"
                    # print(sql)
                    cursor.execute(sql)
                    print(f"All \'{model_type}\' type data has been truncated.")
                    return True

                except Exception as e:
                    print(e)
                    return False
        else:
            print('Truncate Canceled.')
            return False


class AgingModel(MlModel):

    model_type = 'aging_model'

    def register_model(self, dict_info):
        """
        To register new model into ml_model table
        :param dict_info:
            key(1): training_set_id: Integer
            key(2): staff_id : Integer
            key(3): num_of_data : Integer
            key(4_: saved_path : String
        :return: boolean
        """
        return self._db_register_model(self.model_type, dict_info)

    def deregister_model(self, model_id):
        """
        To delete a record from ml_model table
        :param model_id: Integer
        :return: boolean
        """
        return self._db_deregister_model(self.model_type, model_id)

    def retrieve_models(self, dict_condition=None):
        """
        To retrieve models with given dictionary-formed condition
        :param dict_condition:
            key(1): staff_id: Integer
            key(2): start_date: String with "YYYYMMDD" format
            key(3): end_date: String with "YYYYMMDD" format
            key(4): duration : one of ["a week", "a month"]
                    " key(4) must come along with start_date, without end_date "
        :return:list of dictionaries
        """

        if dict_condition is None:
            return self._db_retrieve_models(self.model_type)

        staff_id = None
        start_date, end_date = '20000101', '20991231'
        duration = ''

        for i in list(dict_condition):

            if i is 'staff_id':
                staff_id = dict_condition[i]

            elif i is 'start_date':
                start_date = dict_condition[i]

            elif i is 'end_date':
                end_date = dict_condition[i]

            elif i is 'duration':
                duration = dict_condition[i]

            else:
                return []

        start_date = datetime.strptime(start_date, "%Y%m%d")
        end_date = datetime.strptime(end_date, "%Y%m%d") + timedelta(days=1)

        # duration must come along with start_date, without end_date
        # if start_date is not default value
        if start_date is not '20000101':

            if duration == 'a week':
                end_date = start_date + timedelta(days=8)

            elif duration == 'a month':
                end_date = start_date + relativedelta(months=1) + timedelta(days=1)

            elif duration == '':
                pass

            else:
                print("Invalid Duration input. Input must be \'a week\' or \'a month\'.")
                return []

        # Result will be several rows.
        if staff_id is not None and duration is None:
            condition = f'released_date >= \'{start_date}\' last_updated_date < \'{end_date}\''
            return self._db_retrieve_models(model_type=self.model_type, staff_id=staff_id, condition=condition)

        elif staff_id is not None and duration is not None:
            condition = f'released_date >= \'{start_date}\' and last_updated_date < \'{end_date}\''
            return self._db_retrieve_models(model_type=self.model_type, staff_id=staff_id, condition=condition)

        elif staff_id is None:
            condition = f'released_date >= \'{start_date}\' and last_updated_date < \'{end_date}\''
            return self._db_retrieve_models(model_type=self.model_type, condition=condition)

        else:
            print("No input dictionary.")
            return []

    def retrieve_model_by_ids(self, dict_condition=None):
        """
        To retrieve models with given dictionary-formed condition
        :param dict_condition:
            key(1): id: Integer
            key(2): training_set_id: Integer
            key(3): staff_id : Integer
        :return: boolean
        """

        if dict_condition is None:
            return self._db_retrieve_model_by_ids()

        id, t_id, s_id = None, None, None

        for i in list(dict_condition):
            if i is 'id':
                id = dict_condition[i]
            elif i is 'training_set_id':
                t_id = dict_condition[i]
            elif i is 'staff_id':
                s_id = dict_condition[i]
            else:
                print("Please check your key name again.")
                print(f"\'{i}\' is invalid.")
                return []

        # print(id, t_id, s_id)

        return self._db_retrieve_model_by_ids(id=id, training_set_id=t_id, staff_id=s_id)

    def retrieve_latest_model_id(self):
        """

        :return: id of latest model : Integer
        """
        return self._db_retrieve_latest_model_id(self.model_type)

    def update_model(self, model_id, dict_info):
        """

        :param model_id: Integer
        :param dict_info: Any attributes that you want to change
        :return: boolean
        """
        return self._db_update_model(model_id, dict_info)

    def make_dummy_model(self, count=10):
        return self._db_make_dummy_model(self.model_type, count)

    def truncate_model(self):
        return self._db_truncate_model(model_type=self.model_type)


class EmotionModel(MlModel):

    model_type = 'emotion_model'

    def register_model(self, dict_info):
        """

        :param dict_info:
            key(1): training_set_id: Integer
            key(2): staff_id : Integer
            key(3): num_of_data : Integer
            key(4_: saved_path : String
        :return:
        """
        return self._db_register_model(self.model_type, dict_info)

    def deregister_model(self, model_id):
        """

        :param model_id: Integer
        :return:
        """
        return self._db_deregister_model(self.model_type, model_id)

    def retrieve_models(self, dict_condition=None):
        """
            To retrieve models with given dictionary-formed condition
            :param dict_condition:
                key(1): staff_id: Integer
                key(2): start_date: String with "YYYYMMDD" format
                key(3): end_date: String with "YYYYMMDD" format
                key(4): duration : one of ["a week", "a month"]
                        " key(4) must come along with start_date, without end_date "
            :return:list of dictionaries
            """

        if dict_condition is None:
            return self._db_retrieve_models(self.model_type)

        staff_id = None
        start_date, end_date = '20000101', '20991231'
        duration = ''

        for i in list(dict_condition):

            if i is 'staff_id':
                staff_id = dict_condition[i]

            elif i is 'start_date':
                start_date = dict_condition[i]

            elif i is 'end_date':
                end_date = dict_condition[i]

            elif i is 'duration':
                duration = dict_condition[i]

            else:
                return []

        start_date = datetime.strptime(start_date, "%Y%m%d")
        end_date = datetime.strptime(end_date, "%Y%m%d") + timedelta(days=1)

        # duration must come along with start_date, without end_date
        # if start_date is not default value
        if start_date is not '20000101':

            if duration == 'a week':
                end_date = start_date + timedelta(days=7)

            elif duration == 'a month':
                end_date = start_date + relativedelta(months=1) + timedelta(days=1)

            elif duration == '':
                pass

            else:
                print("Invalid Duration input. Input must be \'a week\' or \'a month\'.")
                return []

        # Result will be several rows.
        if staff_id is not None and duration is None:
            condition = f'released_date >= \'{start_date}\' last_updated_date < \'{end_date}\''
            return self._db_retrieve_models(model_type=self.model_type, staff_id=staff_id, condition=condition)

        elif staff_id is not None and duration is not None:
            condition = f'released_date >= \'{start_date}\' and last_updated_date < \'{end_date}\''
            return self._db_retrieve_models(model_type=self.model_type, staff_id=staff_id, condition=condition)

        elif staff_id is None:
            condition = f'released_date >= \'{start_date}\' and last_updated_date < \'{end_date}\''
            return self._db_retrieve_models(model_type=self.model_type, condition=condition)

        else:
            print("No input dictionary.")
            return []

    def retrieve_model_by_ids(self, dict_condition=None):
        """

        :param dict_condition:
            key(1): id: Integer
            key(2): training_set_id: Integer
            key(3): staff_id : Integer
        :return: boolean
        """

        if dict_condition is None:
            return self._db_retrieve_model_by_ids()

        id = None
        t_id = None
        s_id = None

        for i in list(dict_condition):
            if i is 'id':
                id = dict_condition[i]
            elif i is 'training_set_id':
                t_id = dict_condition[i]
            elif i is 'staff_id':
                s_id = dict_condition[i]
            else:
                print("Please check your key name again.")
                print(f"\'{i}\' is invalid.")
                return []

        return self._db_retrieve_model_by_ids(id=id, training_set_id=t_id, staff_id=s_id)

    def retrieve_latest_model_id(self):
        """

        :return: id of latest model : Integer
        """
        return self._db_retrieve_latest_model_id(self.model_type)

    def update_model(self, model_id, dict_info):
        """

        :param model_id: Integer
        :param dict_info: Any attributes that you want to change
        :return: boolean
        """
        return self._db_update_model(model_id, dict_info)

    def make_dummy_model(self, count=10):
        return self._db_make_dummy_model(self.model_type, count)

    def truncate_model(self):
        return self._db_truncate_model(model_type=self.model_type)


class RecommendationModel(MlModel):

    model_type = 'recommendation_model'

    def register_model(self, model_type, dict_info):
        """
        :param model_type:
            String, one of ["aging_model", "emotion_model", "recommendation_model", face_recognition_model"]
        :param dict_info:
            key(1): training_set_id: Integer
            key(2): staff_id : Integer
            key(3): num_of_data : Integer
            key(4): saved_path : String
        :return: boolean
        """
        return self._db_register_model(model_type, dict_info)

    def deregister_model(self, model_type, model_id):
        """
        :param model_type:
            String, one of ["aging_model", "emotion_model", "recommendation_model", face_recognition_model"]
        :param model_id: Integer
        :return: boolean
        """
        return self._db_deregister_model(model_type, model_id)

    def retrieve_models(self, dict_condition=None):
        """
         To retrieve models with given dictionary-formed condition
         :param dict_condition:
             key(1): staff_id: Integer
             key(2): start_date: String with "YYYYMMDD" format
             key(3): end_date: String with "YYYYMMDD" format
             key(4): duration : one of ["a week", "a month"]
                     " key(4) must come along with start_date, without end_date "
         :return:list of dictionaries
         """

        if dict_condition is None:
            return self._db_retrieve_models(self.model_type)

        staff_id = None
        start_date, end_date = '20000101', '20991231'
        duration = ''

        for i in list(dict_condition):

            if i is 'staff_id':
                staff_id = dict_condition[i]

            elif i is 'start_date':
                start_date = dict_condition[i]

            elif i is 'end_date':
                end_date = dict_condition[i]

            elif i is 'duration':
                duration = dict_condition[i]

            else:
                return []
        try:
            start_date = datetime.strptime(start_date, "%Y%m%d")
            end_date = datetime.strptime(end_date, "%Y%m%d") + timedelta(days=1)

        except Exception as e:
            print(e)
            print("ERROR : WC.mlmodel.retrieve_models")
            print("Check your date format. It should be \'YYYYMMDD\'.")
            return []

        # duration must come along with start_date, without end_date
        # if start_date is not default value
        if start_date is not '20000101':

            if duration == 'a week':
                end_date = start_date + timedelta(days=7)

            elif duration == 'a month':
                end_date = start_date + relativedelta(months=1) + timedelta(days=1)

            elif duration == '':
                pass

            else:
                print("Invalid Duration input. Input must be \'a week\' or \'a month\'.")
                return []

        # Result will be several rows.
        if staff_id is not None and duration is None:
            condition = f'released_date >= \'{start_date}\' last_updated_date < \'{end_date}\''
            return self._db_retrieve_models(model_type=self.model_type, staff_id=staff_id, condition=condition)

        elif staff_id is not None and duration is not None:
            condition = f'released_date >= \'{start_date}\' and last_updated_date < \'{end_date}\''
            return self._db_retrieve_models(model_type=self.model_type, staff_id=staff_id, condition=condition)

        elif staff_id is None:
            condition = f'released_date >= \'{start_date}\' and last_updated_date < \'{end_date}\''
            return self._db_retrieve_models(model_type=self.model_type, condition=condition)

        else:
            print("No input dictionary.")
            return []

    def retrieve_model_by_ids(self, dict_condition=None):
        """
        :param dict_condition:
            key(1): id: Integer
            key(2): training_set_id: Integer
            key(3): staff_id : Integer
        :return: boolean
        """

        if dict_condition is None:
            return self._db_retrieve_model_by_ids()

        id = None
        t_id = None
        s_id = None

        for i in list(dict_condition):
            if i is 'id':
                id = dict_condition[i]
            elif i is 'training_set_id':
                t_id = dict_condition[i]
            elif i is 'staff_id':
                s_id = dict_condition[i]
            else:
                print("Please check your key name again.")
                print(f"\'{i}\' is invalid.")
                return []

        return self._db_retrieve_model_by_ids(id=id, training_set_id=t_id, staff_id=s_id)

    def retrieve_latest_model_id(self, model_type):
        """

        :param model_type:
            String, one of ['aging_model', 'emotion_model', 'recommendation_model', 'face_recognition_model']
        :return: id of model: Integer
        """
        return self._db_retrieve_latest_model_id(model_type)

    def update_model(self, model_id, dict_info):
        """
        :param model_id: Integer
        :param dict_info: Any attributes that you want to change
        :return:boolean
        """
        return self._db_update_model(model_id, dict_info)

    def make_dummy_model(self, count=10):
        return self._db_make_dummy_model(self.model_type, count)

    def truncate_model(self):
        return self._db_truncate_model(model_type=self.model_type)


class FaceRecognitionModel(MlModel):

    model_type = 'face_recognition_model'

    def register_model(self, dict_info):
        """

        :param dict_info:
            key(1): training_set_id: Integer
            key(2): staff_id : Integer
            key(3): num_of_data : Integer
            key(4_: saved_path : String
        :return: boolean
        """
        return self._db_register_model(self.model_type, dict_info)

    def deregister_model(self, model_id):
        """

        :param model_id: Integer
        :return:
        """
        return self._db_deregister_model(self.model_type, model_id)

    def retrieve_models(self, dict_condition=None):
        """

        :param dict_condition:
            key(1): staff_id: Integer
            key(2): start_date: String with "YYYYMMDD" format
            key(3): end_date: String with "YYYYMMDD" format
            key(4): duration : one of ["a week", "a month"]
                    key(4) must come along with start_date, without end_date

        :return:
        """

        if dict_condition is None:
            return self._db_retrieve_models(self.model_type)

        staff_id = None
        start_date = '20000101'
        end_date = '20991231'
        duration = ''

        for i in list(dict_condition):

            if i is 'staff_id':
                staff_id = dict_condition[i]

            elif i is 'start_date':
                start_date = dict_condition[i]

            elif i is 'end_date':
                end_date = dict_condition[i]

            elif i is 'duration':
                duration = dict_condition[i]

            else:
                return []

        start_date = datetime.strptime(start_date, "%Y%m%d")
        end_date = datetime.strptime(end_date, "%Y%m%d") + timedelta(days=1)

        # duration must come along with start_date, without end_date
        # if start_date is not default value
        if start_date is not '20000101':

            if duration == 'a week':
                end_date = start_date + timedelta(days=7)

            elif duration == 'a month':
                end_date = start_date + relativedelta(months=1) + timedelta(days=1)

            elif duration == '':
                pass

            else:
                print("Invalid Duration input. Input must be \'a week\' or \'a month\'.")
                return []

        # Result will be several rows.
        if staff_id is not None and duration is None:
            condition = f'released_date >= \'{start_date}\' last_updated_date < \'{end_date}\''
            return self._db_retrieve_models(model_type=self.model_type, staff_id=staff_id, condition=condition)

        elif staff_id is not None and duration is not None:
            condition = f'released_date >= \'{start_date}\' and last_updated_date < \'{end_date}\''
            return self._db_retrieve_models(model_type=self.model_type, staff_id=staff_id, condition=condition)

        elif staff_id is None:
            condition = f'released_date >= \'{start_date}\' and last_updated_date < \'{end_date}\''
            return self._db_retrieve_models(model_type=self.model_type, condition=condition)

        else:
            print("No input dictionary.")
            return []
        ''

    def retrieve_model_by_ids(self, dict_condition=None):
        """

        :param dict_condition:
            key(1): id: Integer
            key(2): training_set_id: Integer
            key(3): staff_id : Integer
        :return: boolean
        """

        if dict_condition is None:
            return self._db_retrieve_model_by_ids()

        id = None
        t_id = None
        s_id = None

        for i in list(dict_condition):
            if i is 'id':
                id = dict_condition[i]
            elif i is 'training_set_id':
                t_id = dict_condition[i]
            elif i is 'staff_id':
                s_id = dict_condition[i]
            else:
                print("Please check your key name again.")
                print(f"\'{i}\' is invalid.")
                return []

        return self._db_retrieve_model_by_ids(id=id, training_set_id=t_id, staff_id=s_id)

    def retrieve_latest_model_id(self):
        """

        :return: id of latest model : Integer
        """
        return self._db_retrieve_latest_model_id(self.model_type)

    def update_model(self, model_id, dict_info):
        """

        :param model_id: Integer
        :param dict_info: Any attributes that you want to change
        :return:boolean
        """
        return self._db_update_model(self.model_type, model_id, dict_info)

    def make_dummy_model(self, count=10):
        return self._db_make_dummy_model(self.model_type, count)

    def truncate_model(self):
        return self._db_truncate_model(model_type=self.model_type)


if __name__ == '__main__':
    am = AgingModel()
    em = EmotionModel()
    ar = RecommendationModel()
    fr = FaceRecognitionModel()

    # am.retrieve_latest_model_id()[0]['id']
    # em.retrieve_latest_model_id()[0]['id']
    # ar.retrieve_latest_model_id()[0]['id']
    # fr.retrieve_latest_model_id()[0]['id']

    # am.register_model({"training_set_id": 1, "staff_id": 1, "num_of_data": 300, "saved_path": "/root"})
    # em.register_model({"training_set_id": 1, "staff_id": 2, "num_of_data": 400, "saved_path": "/root"})

    # ar.register_model('emotion_model', {"training_set_id": 2, "staff_id": 3, "num_of_data": 500, "saved_path": "/root"})

    # fr.register_model({"training_set_id": 4, "staff_id": 5, "num_of_data": 500, "saved_path": "/root"})

    # am.deregister_model(6)
    # em.deregister_model(3)
    # ar.deregister_model(2)
    # fr.deregister_model(1)

    # print(am.retrieve_models({'staff_id': 1}))
    # print(ar.retrieve_models({'start_date': '20190710', 'duration': 'a week'}))
    # print(ar.retrieve_models({'start_date': '19940915', 'duration': 'a week'}))


    # print(em.retrieve_models({'staff_id': 1}))
    # print(em.retrieve_models({'start_date': '20181103'}))

    # am.retrieve_model_by_ids({"id": 101, "staff_id": 3})
    # am.retrieve_model_by_ids({"id": 101, "training_set_id": 3})
    # am.retrieve_model_by_ids({"id": 101, "staff_id": 3, "training_set_id": 3})

    # em.retrieve_model_by_ids({"staff_id": 3})
    # ar.retrieve_model_by_ids({"id": 101})

    # fr.retrieve_model_by_ids({"id": 2})

    # am.retrieve_models({'staff_id': 1})
    # ar.retrieve_models({'staff_id': 1})

    # am.retrieve_latest_model_id()
    # em.retrieve_latest_model_id()

    # print(type(ar.retrieve_latest_model_id('error_aging_model')))

    # fr.retrieve_latest_model_id()

    # am.update_model(101, {"accuracy": 88})  # error code
    # am.update_model(100, {"accuracdy": '88'})
    am.update_model(101, {"accuracy": '88'})

    # am.make_dummy_model()
    # em.make_dummy_model()
    # ar.make_dummy_model()
    # fr.make_dummy_model()

    # am.truncate_model()
    # em.truncate_model()
    # ar.truncate_model()
    # fr.truncate_model()
