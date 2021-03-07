import WC.dbconnector as db
import WC.stringmaker as sm
import random
from datetime import datetime


class TrainingSet:

    def __init__(self):
        self.conn = None
        db.connect_to_db(self)

    def __del__(self):
        db.disconnect_from_db(self)

    def register_training_set(self, dict_info):
        """

        :param dict_info
            key(1): session_id : Integer
            key(2): saved_path : String
            key(3): num_of_data
        :return:
        """
        return self._db_register_training_set(dict_info)

    def deregister_training_set(self, dict_info):
        """

        :param training_set_id: Integer
        :return: boolean
        """
        return self._db_deregister_training_set(dict_info)

    def retrieve_training_set(self, dict_condition=None):
        """

        :param dict_condition:
            key(1) training_set_id: Integer
            key(2) session_id: Integer
            key(3) start_date: String with "YYYYMMDD" format
            key(4) end_date: String with "YYYYMMDD" format
        :return: List of Dictionaries
        """
        training_set_id = None
        session_id = None
        start_date = '20000101'
        end_date = '20991231'

        if dict_condition is None:
            dict_condition = dict.fromkeys(['training_set_id', 'session_id'])

        for i in list(dict_condition):

            if i is 'training_set_id':
                training_set_id = dict_condition[i]
            elif i is 'session_id':
                session_id = dict_condition[i]
            elif i is 'start_date':
                start_date = dict_condition[i]
            elif i is 'end_date':
                end_date = dict_condition[i]

        start_date = datetime.strptime(start_date, "%Y%m%d")
        end_date = datetime.strptime(end_date, "%Y%m%d")

        duration = f'added_date >= \'{start_date}\' AND added_date <= \'{end_date}\''

        if session_id is not None and training_set_id is not None:
            return self._db_retrieve_training_set(traing_set_id=training_set_id, session_id=session_id, condition=duration)

        elif session_id is not None and training_set_id is None:
            return self._db_retrieve_training_set(session_id=session_id, condition=duration)

        elif session_id is None and training_set_id is not None:
            return self._db_retrieve_training_set(training_set_id=training_set_id, condition=duration)

        else:
            return self._db_retrieve_training_set(condition=duration)

    def make_dummy_training_set(self, count=20):
        return self._db_make_dummy_training_set(count)

    def update_training_set(self, training_set_id, session_id, dict_info):
        """

        :param model_id: Integer
        :param session_id: Integer
        :param dict_info: Any attributes that you want to change
        :return: boolean
        """

        return self._db_update_training_set(training_set_id, session_id, dict_info)

    def truncate_training_set(self):
        return self._db_truncate_training_set()

    def _db_register_training_set(self, dict_info):
        """

        :param dict_info
            key(1): session_id : Integer
            key(2): saved_path : String
            key(3): num_of_data
        :return:
        """

        for i in list(dict_info):
            if i is 'session_id':
                session_id = dict_info[i]
            elif i is 'saved_path':
                saved_path = dict_info[i]
            elif i is 'num_of_data':
                num_of_data = dict_info[i]
        try:
            sql = f"INSERT INTO smart_mirror_system.training_set (s_id, saved_path, added_date, num_of_data) " \
                  f"VALUES (\'{session_id}\', \'{saved_path}\', \'{datetime.now()}\', \'{num_of_data}\')"

        except UnboundLocalError:
            print("You should check your key spelling. Key should be one of ['session_id','saved_path', 'num_of_data']")
            return False

        db.connect_to_db(self)
        with self.conn.cursor() as cursor:
            try:
                cursor.execute(sql)
                db.disconnect_from_db(self)
                print("Training Set has been registered.")
                return True

            except Exception as e:
                print(e)
                print("ERROR : __db_register_training_set")
                db.disconnect_from_db(self)
                return False

    def _db_deregister_training_set(self, dict_info):
        """

        :param dict_info:
            key(1): training_set_id: Integer
            key(2): session_id: Integer
        :return:
        """
        for i in list(dict_info):
            if i is 'training_set_id':
                training_set_id = dict_info[i]
            elif i is 'session_id':
                session_id = dict_info[i]

        if training_set_id and session_id:
            sql = f"DELETE FROM smart_mirror_system.training_set " \
                  f"WHERE id = \'{training_set_id}\' AND s_id = \'{session_id}\'"

        else:
            print("Some parameters are missing")

        with self.conn.cursor() as cursor:
            try:
                # print(sql)
                cursor.execute(sql)
                print("The training set has been deleted.")
                return True

            except Exception as e:
                print(e)
                print("ERROR : __db_deregister_training_set")
                return False

    def _db_retrieve_training_set(self, training_set_id=None, session_id=None, condition=''):
        """
        To get List of dictionaries of training set info
        :param training_set_id: Integer : photo's id
        :param session_id: Integer
        :param condition: partial sql statement about duration
        :return: list of dictionaries
        """

        if training_set_id is not None and session_id is not None:
            with self.conn.cursor() as cursor:
                sql = f'SELECT * FROM smart_mirror_system.training_set ' \
                      f'WHERE s_id = \'{session_id}\' AND training_set_id = \'{training_set_id}\' AND'
                cursor.execute(sql + str(condition))
                result = cursor.fetchall()

                if result is ():
                    return []
                else:
                    return result

        elif training_set_id is None and session_id is not None:
            with self.conn.cursor() as cursor:
                sql = f'SELECT * FROM smart_mirror_system.training_set WHERE s_id = {session_id} AND '
                cursor.execute(sql + str(condition))
                result = cursor.fetchall()

                if result is ():
                    return []
                else:
                    return result

        elif training_set_id is not None and session_id is None:
            with self.conn.cursor() as cursor:
                sql = f'SELECT * FROM smart_mirror_system.training_set WHERE id = {training_set_id} AND '
                cursor.execute(sql)
                result = cursor.fetchall()

                if result is ():
                    return []
                else:
                    return result

        elif training_set_id is None and session_id is None:
            with self.conn.cursor() as cursor:
                sql = f'SELECT * FROM smart_mirror_system.training_set WHERE '
                # print(sql+condition)
                cursor.execute(sql + str(condition))
                result = cursor.fetchall()

                if result is ():
                    return []
                else:
                    return result

    def _db_make_dummy_training_set(self, count):
        with self.conn.cursor() as cursor:
            for _ in range(count):
                try:
                    sql = f"INSERT INTO smart_mirror_system.training_set (s_id, saved_path, added_date, num_of_data) " \
                          f"VALUES (\'{random.randrange(2,10)}\', \'{'/root' +str(random.randrange(1,9999))}\', " \
                          f"\'{datetime.now()}\', \'{random.randrange(1000,9999)}\')"
                    cursor.execute(sql)
                    # print(sql)
                    print("Feedback data has been inserted.")

                except Exception as e:
                    print(e)
                    print('ERROR: _db_make_dummy')
                    return False

            return True

    def _db_update_training_set(self, training_set_id, session_id, dict_info):

        try:
            sql = f"UPDATE smart_mirror_system.training_set SET {sm.StringMaker().key_equation_value(dict_info)} " \
                  f"WHERE id = {training_set_id} and s_id = \'{session_id}\'"

        except TypeError:
            print("")

        with self.conn.cursor() as cursor:
            try:
                cursor.execute(sql)
                print(f"Training set has been updated.")
                return True

            except Exception as e:
                print(e)
                return False

    def _db_truncate_training_set(self):

        option = input('<STAFF ONLY> This method will delete entire rows of the table. Are you Sure? (Y/N)')

        if option.lower() == 'y':
            db.connect_to_db(self)
            with self.conn.cursor() as cursor:
                try:
                    sql = f"TRUNCATE TABLE smart_mirror_system.training_set"
                    cursor.execute(sql)
                    db.disconnect_from_db(self)
                    return True

                except Exception as e:
                    print(e)
                    db.disconnect_from_db(self)
                    return False
        else:
            print('Truncate Canceled.')
            return False


if __name__ == '__main__':

    ts = TrainingSet()
    # ts.register_training_set({'session_id': 1, 'saved_path': '/test', 'num_of_data': 10_000})

    # ts.deregister_training_set({'training_set_id': 2, 'session_id': 1})

    # ts.retrieve_training_set()
    print(ts.retrieve_training_set({'session_id': 7}))
    print(ts.retrieve_training_set({'session_id': 7, 'traing_set_id': 3}))


    # ts.make_dummy_training_set()
    # ts.update_training_set(training_set_id=2, session_id=1, dict_info={"num_of_data": 1818})
