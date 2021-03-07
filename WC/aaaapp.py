import WC.dbconnector as db
import random
from datetime import datetime


class AAAApp:
    def __init__(self):
        self.conn = None

    def register_app(self, dict_info):
        """

        :param dict_info: {"version" : String, "saved_path" : String}
        :return:
        """
        return self._db_register_app(dict_info)

    def deregister_app(self, app_id):
        """

        :param app_id: Integer
        :return: Boolean
        """
        return self._db_deregister_app(app_id)

    def retrieve_app(self, dict_condition=None):
        """
        :param dict_condition
            key(1): Version float
            key(2): start_date
            key(3): released_date
        """
        return self._db_retrieve_app(dict_condition)

    def retrieve_latest_app_id(self):
        """
        To get latest app id
        :return: app_id : Integer
        """

        result = self._db_retrieve_latest_app_id()

        if result:
            return result
        else:
            return False

    def update_app(self, dict_condition):
        """

        :param dict_condition: {"id": Integer, "version': String, "saved_path": String}
        :return:
        """
        id = None
        version = None
        saved_path = None

        for i in list(dict_condition):
            if i is 'id':
                id = dict_condition[i]

            elif i is 'version':
                version = dict_condition[i]

            elif i is 'saved_path':
                saved_path = dict_condition[i]

        return self._db_update_app(app_id=id, version=version, saved_path=saved_path)

    def truncate_app(self):
        return self._db_truncate_app()

    def _db_register_app(self, dict_info):

        version = None
        saved_path = None

        for i in list(dict_info):
            if i is 'version':
                version = dict_info[i]
            elif i is 'saved_path':
                saved_path = dict_info[i]

        sql = f"INSERT INTO aaa_app (version, released_date, saved_path) VALUES (\'{version}\',\'{datetime.now()}\', \'{saved_path}\')"

        db.connect_to_db(self)
        with self.conn.cursor() as cursor:
            try:
                cursor.execute(sql)
                db.disconnect_from_db(self)
                return True

            except Exception as e:
                print(e)
                print("ERROR : _db_register_app")
                db.disconnect_from_db(self)
                return False

    def _db_deregister_app(self, app_id):

        sql = f"DELETE FROM aaa_app WHERE id = \'{app_id}\'"

        db.connect_to_db(self)
        with self.conn.cursor() as cursor:
            try:
                cursor.execute(sql)
                db.disconnect_from_db(self)
                return True

            except Exception as e:
                print(e)
                print("ERROR : _db_deregister_app")
                db.disconnect_from_db(self)
                return False

    def _db_retrieve_app(self, dict_condition):
        """

        :param dict_condition
            key(1): Version: float
            key(2): start_date: String with "YYYYMMDD" foramt
            key(3): end_date: String with "YYYYMMDD" foramt
        """

        version = None
        start_date = '20000101'
        end_date = '20991231'

        if dict_condition is None:
            dict_condition = dict.fromkeys(['version', 'start_date', 'end_date'])
            dict_condition.update({'start_date': '20000101', 'end_date': '20991231'})

        else:
            for i in list(dict_condition):

                if i is 'version':
                    version = dict_condition[i]
                elif i is 'start_date':
                    start_date = dict_condition[i]
                elif i is 'end_date':
                    end_date = dict_condition[i]

        start_date = datetime.strptime(start_date, "%Y%m%d")
        end_date = datetime.strptime(end_date, "%Y%m%d")

        duration = f'released_date >= \'{start_date}\' AND released_date <= \'{end_date}\''
        if version is not None:
            sql = f"SELECT * FROM smart_mirror_system.aaa_app WHERE version = \'{version}\' AND "
        elif version is None:
            sql = f"SELECT * FROM smart_mirror_system.aaa_app WHERE "

        db.connect_to_db(self)
        with self.conn.cursor() as cursor:
            try:
                # print(sql+duration)
                cursor.execute(sql+duration)

                result = cursor.fetchall()
                if result is ():
                    return []
                else:
                    db.disconnect_from_db(self)
                    return result

            except Exception as e:
                print(e)
                print('ERROR : _db_retrieve_app')
                db.disconnect_from_db(self)
                return False

    def _db_retrieve_latest_app_id(self):

        sql = 'SELECT id FROM smart_mirror_system.aaa_app ORDER BY version DESC limit 1'

        db.connect_to_db(self)
        with self.conn.cursor() as cursor:
            try:
                cursor.execute(sql)
                result = cursor.fetchall()
                db.disconnect_from_db(self)
                if result is():
                    return []
                else:
                    return result[0]['id']

            except Exception as e:
                print(e)
                print("ERROR : __db_retrieve_latest_app_id")
                db.disconnect_from_db(self)
                return False

    def _db_update_app(self, app_id, version, saved_path):

        sql_top = f"UPDATE aaa_app "
        sql_bot = f", released_date = \'{datetime.now()}\' WHERE id = \'{app_id}\'"

        if version is not None and saved_path is not None:
            condition = f"SET version = \'{version}\', saved_path = \'{saved_path}\' "
        elif version is not None and saved_path is None:
            condition = f'SET version = \'{version}\' '
        elif version is None and saved_path is not None:
            condition = f'SET saved_path = \'{saved_path}\' '

        db.connect_to_db(self)
        with self.conn.cursor() as cursor:
            try:
                cursor.execute(sql_top + condition + sql_bot)
                db.disconnect_from_db(self)
                return True

            except Exception as e:
                print(e)
                print("ERROR : __db_update_app")
                db.disconnect_from_db(self)
                return False

    def _db_truncate_app(self):

        option = input('<STAFF ONLY> This method will delete entire rows of the table. Are you Sure? (Y/N)')

        if option.lower() == 'y':
            db.connect_to_db(self)
            with self.conn.cursor() as cursor:
                try:
                    sql = f"TRUNCATE TABLE smart_mirror_system.aaa_app"
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
    ap = AAAApp()
    # ap.register_app({"version": 1.0, "saved_path": '/root'})

    # ap.retrieve_app()
    # ap.update_app({"id": 1, "version": 1.1, "saved_path": '/test'})
    # ap.deregister_app(app_id=1)
    # ap.truncate_app()
    # print(ap.retrieve_latest_app_id())
