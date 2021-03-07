import dbconnector as db
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
        return self.__db_register_app(dict_info)

    def deregister_app(self, app_id):
        """

        :param app_id: Integer
        :return: Boolean
        """
        return self.__db_deregister_app(app_id)

    # TODO : retrieve all
    def retrieve_app(self):
        pass

    def retrieve_latest_app_id(self):
        """
        To get latest app id
        :return: app_id : Integer
        """

        result = self.__db_retrieve_latest_app_id()

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

        return self.__db_update_app(app_id=id, version=version, saved_path=saved_path)

    def truncate_app(self):
        return self.__db_truncate_app()

    def __db_register_app(self, dict_info):

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
                print("ERROR : __db_register_app")
                db.disconnect_from_db(self)
                return False

    def __db_deregister_app(self, app_id):

        sql = f"DELETE FROM aaa_app WHERE id = \'{app_id}\'"

        db.connect_to_db(self)
        with self.conn.cursor() as cursor:
            try:
                cursor.execute(sql)
                db.disconnect_from_db(self)
                return True

            except Exception as e:
                print(e)
                print("ERROR : __db_deregister_app")
                db.disconnect_from_db(self)
                return False

    def __db_retrieve_latest_app_id(self):

        sql = 'SELECT id FROM smart_mirror_system.aaa_app ORDER BY version DESC limit 1'

        db.connect_to_db(self)
        with self.conn.cursor() as cursor:
            try:
                cursor.execute(sql)
                result = cursor.fetchall()
                db.disconnect_from_db(self)

                return result

            except Exception as e:
                print(e)
                print("ERROR : __db_retrieve_latest_app_id")
                db.disconnect_from_db(self)
                return False

    def __db_update_app(self, app_id, version, saved_path):

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

    def __db_truncate_app(self):

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

# ap = AAAApp()

# ap.register_app({"version": 1.0, "saved_path": '/root'})
# ap.update_app({"id": 1, "version": 1.1, "saved_path": '/test'})
# ap.deregister_app(app_id=1)
# ap.truncate_app()
# ap.retrieve_latest_app_id()[0]['id']