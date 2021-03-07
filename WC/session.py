import WC.dbconnector as db
import WC.dummymaker as dm
from datetime import datetime
from dateutil.relativedelta import relativedelta
import random
from datetime import timedelta


class Session:

    def __init__(self):
        self.conn = None

    def start_session(self, user_id):
        """
        To start session when user has logged in.
        :param user_id: Integer
        :return: boolean
        """

        if self._db_register_session(user_id):
            print("Session has started.")
            return True
        else:
            return False

    def finish_session(self, user_id):
        """
        To finish session when user has logged out.
        This method doesn't delete a row of session table.
        Just Fill out log_out_date column with datetime.now()
        :param user_id:
        :return: boolean
        """

        if self._db_update_session(user_id):
            print("Session has finished.")
            return True
        else:
            return False

    def retrieve_sessions(self, dict_condition=None):
        """
        To retrieve rows in session table with given dictionary formed conditions.
        :param dict_condition:
            key(1): user_id: Integer
            key(2): start_date: String with "YYYYMMDD" format
            key(3): end_date: String with "YYYYMMDD" format
            key(4): duration : one of ["a week", "a month"]
                    key(4) must come along with start_date, without end_date

        :return: list of dictionaries
        """

        if dict_condition is None:
            return self._db_retrieve_sessions()

        user_id = None
        start_date = '20000101'
        end_date = '20991231'
        duration = ''

        for i in list(dict_condition):

            if i is 'user_id':
                user_id = dict_condition[i]

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
        if user_id is not None and duration is None:
            condition = f'and log_in_date >= \'{start_date}\' and log_out_date < \'{end_date}\''
            return self._db_retrieve_sessions(user_id=user_id, condition=condition)

        elif user_id is not None and duration is not None:
            condition = f'and log_in_date >= \'{start_date}\' and log_out_date < \'{end_date}\''
            return self._db_retrieve_sessions(user_id=user_id, condition=condition)

        elif user_id is None:
            condition = f'where log_in_date >= \'{start_date}\' and log_out_date < \'{end_date}\''
            return self._db_retrieve_sessions(condition=condition)

        else:
            print("No input dictionary.")
            return []

    def retrieve_active_session_id(self, user_id):
        """
        To get user who matches with given user_id 's session id who are active right now.
        :param user_id: Integer
        :return: session_id : Integer
        """

        result = self._db_retrieve_active_session_id(user_id)

        if result:
            return result

        else:
            print(f'user_id : \'{user_id}\' is not currently logged in.')
            return []

    def update_action_log(self, user_id, action):
        """

        :param user_id: Integer
        :param action: String , one of ['Greeting', 'Analyze Aging', 'Analyze Emotion', 'Recommend Aging',
                                        'Take Photo', 'Generate Report', 'Feedback', 'Change User Info']
        :return: boolean
        """
        return self._db_update_action_log(user_id,action)

    def make_dummy_sessions(self, count=200):
        """
        For testing, make dummy records of session table
        :param count: Integer, default value = 200
        :return: boolean
        """
        return self._db_make_dummy_sessions(count)

    def truncate_session(self):
        """
        For testing, delete all rows of session table
        :return: boolean
        """

        option = input('<STAFF ONLY> This method will delete entire rows of the table. Are you Sure? (Y/N)').lower()
        if option == 'y':
            if self._db_truncate_session():
                print("The table has been truncated.")
                return True
            else:
                return False

        elif option == 'n':
            return False

        else:
            print("Invalid input.")
            self.truncate_session()
            return

    def _db_register_session(self, user_id):

        db.connect_to_db(self)
        with self.conn.cursor() as cursor:

            # To check unfinished session already exists
            sql = f"SELECT * FROM smart_mirror_system.session where u_id = {user_id} and log_out_date is null"
            cursor.execute(sql)
            result = cursor.fetchone()

            if result:
                db.disconnect_from_db(self)
                print("This ID is already being accessed. Disconnect the existing connection and reconnect.")
                self._db_update_session(user_id)
                self._db_register_session(user_id)
                return True

            else:
                try:
                    sql = "INSERT INTO session (u_id,log_in_date) VALUES(%s, %s)"
                    cursor.execute(sql, (user_id, datetime.now()))
                    db.disconnect_from_db(self)
                    return True

                except Exception as e:
                    print(e)
                    db.disconnect_from_db(self)
                    return False

    def _db_retrieve_sessions(self, user_id=None, condition=""):
        db.connect_to_db(self)
        with self.conn.cursor() as cursor:

            try:
                if user_id:
                    sql = f'SELECT * FROM smart_mirror_system.session where u_id = {user_id} '

                else:
                    sql = f'SELECT * FROM smart_mirror_system.session '

                # print('testing sql (line 187) :\n'+ sql + str(condition))
                cursor.execute(sql + str(condition))
                result = cursor.fetchall()
                db.disconnect_from_db(self)
                if result is ():
                    return []
                else:
                    return result

            except Exception as e:
                print(e)
                print("_db_retrieve_session: error")
                db.disconnect_from_db(self)
                return []

    def _db_retrieve_active_session_id(self, user_id):

            db.connect_to_db(self)
            with self.conn.cursor() as cursor:
                try:
                    sql = f'SELECT id FROM smart_mirror_system.session WHERE u_id = {user_id} AND log_out_date is null'
                    cursor.execute(sql)
                    result = cursor.fetchall()
                    db.disconnect_from_db(self)
                    if result is ():
                        return []
                    else:
                        return result[0]['id']

                except Exception as e:
                    print(e)
                    db.disconnect_from_db(self)
                    return []

    def _db_update_session(self, user_id):
        db.connect_to_db(self)
        with self.conn.cursor() as cursor:

            try:
                # To check if log_out_date is NULL
                sql = f"SELECT * FROM session WHERE u_id = {user_id} and log_out_date is null"
                cursor.execute(sql)

                if cursor.fetchone():
                    sql = f"UPDATE session SET log_out_date = %s WHERE u_id = {user_id}"
                    cursor.execute(sql, (datetime.now()))
                    db.disconnect_from_db(self)
                    return True

                else:
                    print("This id has been already logged out.")
                    return False

            except Exception as e:
                print(e)
                db.disconnect_from_db(self)
                return False

    # # test code 1
    # def _db_update_action_log1(self, session_id, action):
    #     """
    #
    #     :param session_id: Integer
    #     :param action:
    #         String, one of [Greeting, analyze Aging, analyze Emotion, Recommend Aging,
    #                        Take Photo, Generate Report, Feedback, Change User Info]
    #     :return:
    #     """
    #
    #     sql_get_action_log = f"SELECT action_log FROM smart_mirror_system.session WHERE id = \'{session_id}\'"
    #
    #     db.connect_to_db(self)
    #     with self.conn.cursor() as cursor:
    #         try:
    #             cursor.execute(sql_get_action_log)
    #             result = cursor.fetchall()
    #             if result is ():
    #                 print(f"There is no matching data with session id : \'{session_id}\'")
    #                 return False
    #
    #             else:
    #                 action_log = result[0]['action_log']
    #                 print(type(action_log))
    #                 print(action_log)
    #                 action_log = action_log + ', ' + action
    #         except Exception as e:
    #             print(e)
    #             db.disconnect_from_db(self)
    #             return False
    #
    #     update_action_log = f"UPDATE smart_mirror_system.session SET action_log = \'{action_log}\' WHERE id = \'{session_id}\'"

    def _db_update_action_log(self, user_id, action):

        session_id = self.retrieve_active_session_id(user_id)

        # print(session_id)
        if session_id:

            action = action + ', '

            sql = f'UPDATE smart_mirror_system.session ' \
                  f'SET action_log = CONCAT_WS(IFNULL(action_log,\'\'), \'{action}\') ' \
                  f'WHERE id = \'{session_id}\' AND log_out_date is Null'

            db.connect_to_db(self)
            with self.conn.cursor() as cursor:
                try:
                    cursor.execute(sql)
                    db.disconnect_from_db(self)
                    print(f"Session id: \'{session_id}\' \'s Action log has been updated")
                    return True

                except Exception as e:
                    print(e)
                    print("ERROR : _db_update_action_log")
                    db.disconnect_from_db(self)
                    return False

        else:
            return False

    def _db_make_dummy_sessions(self, count):

        db.connect_to_db(self)
        with self.conn.cursor() as cursor:
            for i in range(count):
                log_in_date = datetime.strptime(dm.dummy_date(start_year=2018), "%Y%m%d%H")
                random_hours = random.randrange(1, 24)
                random_minutes = random.randrange(0, 60)
                log_out_date = log_in_date + timedelta(hours=random_hours) + timedelta(minutes=random_minutes)
                try:
                    sql = f'INSERT INTO smart_mirror_system.session(u_id, log_in_date, log_out_date)' \
                        f' VALUES(\'{random.randrange(1,100)}\', \'{log_in_date}\', \'{log_out_date}\')'
                    # print(sql)
                    cursor.execute(sql)

                except Exception as e:
                    print(e)
                    db.disconnect_from_db(self)
                    return False

    def _db_truncate_session(self):

        db.connect_to_db(self)
        with self.conn.cursor() as cursor:
            try:
                sql = "TRUNCATE TABLE smart_mirror_system.session;"
                cursor.execute(sql)
                db.disconnect_from_db(self)
                return True

            except Exception as e:
                print(e)
                db.disconnect_from_db(self)
                return False


if __name__ == '__main__':
    ss = Session()
    # ss.start_session(293)
    # ss.retrieve_sessions({'user_id': 95, 'start_date': '20180228', 'duration': 'a month'})
    # print(ss.retrieve_sessions({'user_id': 3}))
    # ss.retrieve_sessions({'user_id': 95, 'start_date': '20180228', 'duration': 'a week'})
    # ss.retrieve_sessions({'user_id': 95, 'start_date': '20180228', 'end_date': '20190708'})
    # ss.retrieve_sessions({'start_date': '20180228'})

    # ss.retrieve_active_session_id(295)

    # ss._db_update_action_log(293, 'Feedback')

    # ss.finish_session(293)
    # ss.get_session_id(1)
    #
    # ss.make_dummy_sessions()

    # print(ss.retrieve_active_session_id(1))
