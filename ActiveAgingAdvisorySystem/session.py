import WC.dbconnector as db
import WC.dummymaker as dm
from datetime import datetime
import random
from datetime import timedelta


class Session:

    def __init__(self):
        self.conn = None

    def start_session(self, user_id):

        if self.__db_register_session(user_id):
            print("Session has started.")
            return True
        else:
            return False

    def finish_session(self, user_id):

        if self.__db_update_session(user_id):
            print("Session has finished.")
            return True
        else:
            return False

    def truncate_session(self):
        option = input('<STAFF ONLY> This method will delete entire rows of the table. Are you Sure? (Y/N)').lower()
        if option == 'y':
            if self.__db_truncate_session():
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

    def retrieve_session(self, dict_condition=None):

        if dict_condition is None:
            return self.__db_retrieve_session()

        session_id = None
        user_id = None
        start_date = '20000101'
        end_date = '20991231'

        for i in list(dict_condition):

            if i is 'session_id':
                session_id = dict_condition[i]

            elif i is 'user_id':
                user_id = dict_condition[i]

            elif i is 'start_date':
                start_date = dict_condition[i]

            elif i is 'end_date':
                end_date = dict_condition[i]

            else:
                return []

        # Result will be one row.
        if session_id:
            return self.__db_retrieve_session(session_id=session_id)

        # Result will be several rows.
        elif user_id:
            condition = f'and log_in_date >= \'{start_date}\' and log_out_date <= \'{end_date}\''
            return self.__db_retrieve_session(user_id=user_id, condition=condition)

        else:
            print("No input dictionary.")
            return []

    def retrieve_active_session_id(self, user_id):
        """
        To get user's session id who are active right now.
        :param user_id: Integer
        :return: session_id : Integer
        """

        result = self.__db_retrieve_active_session_id(user_id)

        if result:
            return result['id']

        else:
            print(f'user_id : \'{user_id}\' is not currently logged in.')
            return []

    def make_dummy_session(self, count=200):

        db.connect_to_db(self)
        with self.conn.cursor() as cursor:
            for i in range(count):
                log_in_date = datetime.strptime(dm.dummy_date(start_year=2018), "%Y%m%d%H")
                random_hours = random.randrange(1, 24)
                random_minutes = random.randrange(0, 60)
                log_out_date =  log_in_date + timedelta(hours=random_hours) + timedelta(minutes=random_minutes)
                try:
                    sql = f'INSERT INTO session(u_id, log_in_date, log_out_date)' \
                        f' VALUES(\'{random.randrange(1,100)}\', \'{log_in_date}\', \'{log_out_date}\')'
                    print(sql)
                    cursor.execute(sql)

                except Exception as e:
                    print(e)
                    db.disconnect_from_db(self)
                    return False

    def __db_register_session(self, user_id):

        db.connect_to_db(self)
        with self.conn.cursor() as cursor:

            # To check unfinished session already exists
            sql = f"SELECT * FROM session where u_id = {user_id} and log_out_date is null"
            cursor.execute(sql)
            result = cursor.fetchone()

            if result:
                db.disconnect_from_db(self)
                print("This ID is already being accessed. Disconnect the existing connection and reconnect.")
                self.__db_update_session(user_id)
                self.__db_register_session(user_id)
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

    def __db_retrieve_session(self, session_id=None, user_id=None, condition=""):
        """
        To find user's Log-in log
        :param user_id:
        :return:
        """

        db.connect_to_db(self)
        with self.conn.cursor() as cursor:

            try:
                if session_id:
                    sql = f"SELECT * FROM session where id = {session_id} "

                elif user_id:
                    sql = f'SELECT * FROM session where u_id = {user_id} '

                else:
                    sql = f'SELECT * FROM session'

                cursor.execute(sql + str(condition))
                result = cursor.fetchall()
                db.disconnect_from_db(self)
                return result

            except Exception as e:
                print(e)
                print("__db_retrieve_session: error")
                db.disconnect_from_db(self)
                return []

    def __db_retrieve_active_session_id(self, user_id):

            db.connect_to_db(self)
            with self.conn.cursor() as cursor:
                try:
                    sql = f'SELECT id FROM session WHERE u_id = {user_id} AND log_out_date is null'
                    cursor.execute(sql)
                    result = cursor.fetchone()
                    db.disconnect_from_db(self)
                    return result

                except Exception as e:
                    print(e)
                    db.disconnect_from_db(self)
                    return None

    def __db_update_session(self, user_id):

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

    def __db_truncate_session(self):

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

# ss = Session()

# ss.start_session(293)
# print(ss.retrieve_session())
#
# print(ss.retrieve_session({"session_id": 6}))
# ss.finish_session(1)
# ss.get_session_id(1)
#
# ss.make_dummy_session()

# print(ss.retrieve_active_session_id(1))