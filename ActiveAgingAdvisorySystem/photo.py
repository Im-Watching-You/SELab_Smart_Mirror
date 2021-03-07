import WC.session as ss
import WC.dbconnector as db
import WC.dummymaker as dm
from datetime import datetime
import random


class Photo:

    def __init__(self):
        self.conn = None
        self.session = ss.Session()
        self.photo_id = 1

    def register_photo(self, dict_info):
        """
        To register photo info into DB
        :param dict_info:
            key(1) user_id : Integer
            key(2) saved_path : String
        :return: Boolean
        """

        # if 'user_id' in dict_info.keys():
        #     return False
        session_id = dict_info['session_id']
        saved_path = dict_info['saved_path']

        if session_id:
            if self._db_register_photo(session_id, saved_path):
                # To manage photo_id
                self.photo_id += 1
                return True
            else:
                return False

        else:
            print("Logged out user or No corresponding session ID.")
            return False

    def deregister_photo(self, dict_info):
        """

        :param dict_info:
            key(1): photo_id : Integer
            key(2): session_id : Integer
        :return: boolean
        """
        # print(session_id)
        return self._db_deregister_photo(dict_info)

    def retrieve_photo(self, dict_condition=None):
        """

        :param dict_condition:
            key(1) photo_id: Integer
            key(2) session_id: Integer
            key(3) start_date: String with "YYYYMMDD" format
            key(4) end_date: String with "YYYYMMDD" format
        :return: List of Dictionaries
        """

        if dict_condition is None:
            dict_condition = dict.fromkeys(['photo_id', 'session_id', 'start_date', 'end_date'])
            dict_condition.update({'start_date': '20000101', 'end_date': '20991231'})

        photo_id = None
        session_id = None
        start_date = '20000101'
        end_date = '20991231'

        for i in list(dict_condition):

            if i is 'photo_id':
                photo_id = dict_condition[i]
            elif i is 'session_id':
                session_id = dict_condition[i]
            elif i is 'start_date':
                start_date = dict_condition[i]
            elif i is 'end_date':
                end_date = dict_condition[i]

        start_date = datetime.strptime(start_date, "%Y%m%d")
        end_date = datetime.strptime(end_date, "%Y%m%d")

        duration = f'taken_date >= \'{start_date}\' AND taken_date <= \'{end_date}\''

        if session_id is not None and photo_id is not None:
            return self._db_retrieve_photo(photo_id=photo_id, session_id=session_id)

        elif session_id is not None and photo_id is None:
            return self._db_retrieve_photo(session_id=session_id, condition=duration)

        elif session_id is None and photo_id is not None:
            return self._db_retrieve_photo(photo_id=photo_id, condition=duration)

        else:
            print(duration)
            return self._db_retrieve_photo(condition=duration)

    def retrieve_latest_photo_id(self, session_id):
        """
        :param session_id: Integer
        :return:
        """
        return self._db_retrieve_latest_photo_id(session_id)

    def update_photo(self, dict_info):
        """
        :param dict_info:
            key(1): photo_id: Integer
            key(2): session_id: Integer
            key(3): saved_path: String
        """
        return self._db_update_photo(dict_info)

    def make_dummy_photo(self, count=100):
        return self._db_make_dummy_photo(count)

    def truncate_photo(self):
        return self._db_truncate_photo()

    def reset_photo_id(self):
        """
        If session has started, set photo_id to zero.
        :return: None
        """
        self.photo_id = 1

    def _db_register_photo(self, session_id, saved_path):

        db.connect_to_db(self)
        with self.conn.cursor() as cursor:
            try:
                sql = f"INSERT INTO photo (id ,s_id, saved_path, taken_date) VALUES(\'{self.photo_id}\', " \
                      f"\'{session_id}\', \'{saved_path}\', \'{datetime.now()}\')"
                cursor.execute(sql)
                # print(sql)
                db.disconnect_from_db(self)
                return True

            except Exception as e:
                print(e)
                db.disconnect_from_db(self)
                return False

    def _db_deregister_photo(self, dict_info):
        """

        :param dict_info:
            key(1): photo_id : Integer
            key(2): session_id : Integer
        :return: boolean
        """
        for i in list(dict_info):
            if i is 'photo_id':
                photo_id = dict_info[i]
            elif i is 'session_id':
                session_id = dict_info[i]

        if photo_id is not None and session_id is not None:
            sql = f"DELETE FROM smart_mirror_system.photo WHERE id = \'{photo_id}\' AND s_id = \'{session_id}\'"
        elif photo_id is None and session_id is not None:
            sql = f"DELETE FROM smart_mirror_system.photo WHERE s_id = \'{session_id}\'"
        else:
            print("ERROR: _db_deregister_photo. Ambiguous Input. Only id cannot specify one photo.")
            print("Or if you didn't input anything, method will do nothing.")
            return False

        sql_count_rows = f"Select count(id) from smart_mirror_system.photo"

        db.connect_to_db(self)

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
                    print(f"There is no matching photo with given ids \n"
                          f"(1) Photo id: \'{photo_id}\' (2) Session id: \'{session_id}\'")

                else:
                    print(f"The photo has been deleted.")

                db.disconnect_from_db(self)
                return True

            except Exception as i:
                print(i)
                print("ERROR: _db_deregister_photo")
                return False

    def _db_retrieve_photo(self, photo_id=None, session_id=None, condition=""):
        """
        To get List of dictionaries of photo info
        :param photo_id: Integer : photo's id
        :param session_id: Integer
        :return:
        """

        db.connect_to_db(self)

        if photo_id is not None and session_id is not None:
            with self.conn.cursor() as cursor:
                sql = f'SELECT * FROM smart_mirror_system.photo WHERE s_id = \'{session_id}\' AND id = \'{photo_id}\' AND'
                # print(sql+str(condition))
                cursor.execute(sql+str(condition))
                result = cursor.fetchall()

                if result is ():
                    return []
                else:
                    return result

        elif photo_id is None and session_id is not None:
            with self.conn.cursor() as cursor:
                sql = f'SELECT * FROM smart_mirror_system.photo WHERE s_id = {session_id} AND '
                # print(sql+str(condition))
                cursor.execute(sql+str(condition))
                result = cursor.fetchall()

                if result is ():
                    return []
                else:
                    return result

        elif photo_id is not None and session_id is None:
            with self.conn.cursor() as cursor:
                sql = f'SELECT * FROM smart_mirror_system.photo WHERE id = {photo_id} AND '
                cursor.execute(sql)
                result = cursor.fetchall()

                if result is ():
                    return []
                else:
                    return result

        elif photo_id is None and session_id is None:

            with self.conn.cursor() as cursor:
                sql = f'SELECT * FROM photo WHERE '
                print(sql+str(condition))
                cursor.execute(sql+str(condition))
                result = cursor.fetchall()

                if result is ():
                    return []
                else:
                    return result

    def _db_retrieve_latest_photo_id(self, session_id):

        sql = f'SELECT * FROM smart_mirror_system.photo WHERE s_id = \'{session_id}\' ORDER BY taken_date DESC limit 1 '

        db.connect_to_db(self)
        with self.conn.cursor() as cursor:
            try:
                # print(sql)
                cursor.execute(sql)
                result = cursor.fetchall()

                if result is ():
                    return -1
                else:
                    return result[0]['id']

            except Exception as e:
                print(e)
                db.disconnect_from_db(self)
                return -1

    def _db_update_photo(self, dict_info):

        photo_id = None
        session_id = None
        saved_path = None

        for i in list(dict_info):
            if i is 'photo_id':
                photo_id = dict_info[i]
            elif i is 'session_id':
                session_id = dict_info[i]
            elif i is 'saved_path':
                saved_path = dict_info[i]

        if photo_id is not None and session_id is not None and saved_path is not None:
            sql = f"UPDATE smart_mirror_system.photo SET saved_path = \'{saved_path}\' " \
                  f"WHERE id = \'{photo_id}\' AND s_id = \'{session_id}\'"
        else:
            print("ERROR: _db_update_photo : Missing Input.")
            return False
        # print(sql)
        db.connect_to_db(self)
        with self.conn.cursor() as cursor:
            try:
                cursor.execute(sql)
                db.disconnect_from_db(self)
                return True

            except Exception as e:
                print(e)
                db.disconnect_from_db(self)
                return False

    def _db_make_dummy_photo(self, count):
        db.connect_to_db(self)
        with self.conn.cursor() as cursor:
            try:
                for i in range(count):
                    saved_path = '/root/' + str(random.randrange(1, 99999))
                    random_date = datetime.strptime(dm.dummy_date(start_year=2018), "%Y%m%d%H")
                    sql = f"INSERT INTO photo (s_id, saved_path, taken_date) VALUES(\'{random.randrange(1,400)}\', " \
                        f"\'{saved_path}\', \'{random_date}\')"
                    cursor.execute(sql)

            except Exception as e:
                print(e)
                db.disconnect_from_db(self)
                return False

            return True

    def _db_truncate_photo(self):
        db.connect_to_db(self)
        with self.conn.cursor() as cursor:
            try:
                sql = "TRUNCATE TABLE smart_mirror_system.photo;"
                cursor.execute(sql)
                db.disconnect_from_db(self)
                return True

            except Exception as e:
                print(e)
                db.disconnect_from_db(self)
                return False


if __name__ == '__main__':
    pt = Photo()
    # 하나의 인스턴스 내의 변수를 id로 이용함.
    # 인스턴스를 새로 생성해서 똑같은 작업을 수행하면 primary key 오류
    # 하나의 인스턴스로 연속적으로  register 하면 문제 없음

    # pt.register_photo({'id': 293, 'saved_path': '/root/test/1'})
    # pt.register_photo({'id': 293, 'saved_path': '/root/test/1'})

    # pt.deregister_photo(id=1)

    # pt.retrieve_photo({"session_id": 468})
    print(pt.retrieve_latest_photo_id(1))

    # pt.retrieve_photo({"session_id": 30, "start_date": '20190101'})
    # pt.update_photo({"photo_id":1, "session_id":350, "saved_path":'/test'})

    # pt.make_dummy_photo()
    # pt.truncate_photo()
