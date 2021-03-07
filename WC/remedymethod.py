import WC.stringmaker as sm
import WC.dbconnector as db
from datetime import datetime


class RemedyMethod:
    def __init__(self):
        self.conn = None
        db.connect_to_db(self)
        self.sm = sm.StringMaker()

    def __del__(self):
        db.disconnect_from_db(self)

    def register_remedy_method(self, dict_info):
        """

        :param dict_info:
        :return:
        """
        return self._db_register_remedy_method(dict_info)

    def deregister_remedy_method(self, remedy_method_id):
        """

        :param remedy_method_id: Integer
        :return:
        """
        return self._db_deregister_remedy_method(remedy_method_id)

    def retrieve_remedy_method_by_id(self, remedy_method_id):
        """

        :param remedy_method_id: Intger
        :return: list of dictionaries
        """
        return self._db_retrieve_remedy_method_by_id(remedy_method_id)

    def retrieve_rm_by_provider(self, dict_condition):
        """

        :param dict_condition:
            key(1) remedy_method_id: Integer
            key(2) symptom: String
            key(3) provider: String
            key(4) tag: String
        :return:
        """
        return self._db_retrieve_remedy_method_by_provider(dict_condition)

    def retrieve_all_rm(self, symptom=None):
        """

        :param symptom: String
        :return:
        """
        return self._db_retrieve_all_rm(symptom)

    def update_remedy_method(self, remedy_method_id, dict_info):
        return self._db_update_remedy_method(remedy_method_id, dict_info)

    def truncate_remedy_method(self):
        pass

    def _db_register_remedy_method(self, dict_info):

        rm_type = None
        symptom = None
        provider = None
        url = None
        maintain, improve, prevent = 0, 0, 0
        description = None
        edit_date = datetime.now()
        tag = None

        is_decided = 0

        for i in list(dict_info):
            if i is 'rm_type':
                rm_type = dict_info[i]
            elif i is 'symptom':
                symptom = dict_info[i]
            elif i is 'provider':
                provider = dict_info[i]
            elif i is 'url':
                url = dict_info[i]

            elif i is 'maintain' and is_decided is 0:
                maintain = 1
                is_decided = 1
                print('Effectiveness : maintain')

            elif i is 'improve' and is_decided is 0:
                improve = 1
                is_decided = 1
                print('Effectiveness : improve')

            elif i is 'prevent' and is_decided is 0:
                prevent = 1
                is_decided = 1
                print('Effectiveness : prevent')

            elif i is 'description':
                description = dict_info[i]
            elif i is 'tag':
                tag = dict_info[i]

        # print(maintain)
        # print(improve)

        sql = f'INSERT INTO remedy_method (rm_type, symptom, provider, url, maintain, improve, prevent, ' \
            f'description, edit_date, tag) VALUES (\'{rm_type}\', \'{symptom}\', \'{provider}\', \'{url}\',' \
            f' \'{maintain}\', \'{improve}\', \'{prevent}\', \'{description}\', \'{edit_date}\', \'{tag}\')'

        # print(sql)

        with self.conn.cursor() as cursor:
            try:
                cursor.execute(sql)
                print("Remedy Method has been inserted.")
                return True

            except Exception as e:
                print(e)
                return False

    def _db_retrieve_remedy_method_by_id(self, rm_id):

        sql = f'SELECT * FROM remedy_method WHERE rm_id = {rm_id}'

        with self.conn.cursor() as cursor:
            try:
                cursor.execute(sql)
                result = cursor.fetchall()

                if result is():
                    return []
                else:
                    return result

            except Exception as e:
                print(e)
                return []

    def _db_retrieve_remedy_method_by_provider(self, dict_condition):
        """

        :param dict_condition:
            key(1) remedy_method_id: Integer
            key(2) symptom: String
            key(3) provider: String
            key(4) tag: String
        :return:
        """
        rm_id, symptom, provider, tag = None, None, None, None

        for i in list(dict_condition):
            if i is 'remedy_method_id':
                rm_id = dict_condition[i]
            elif i is 'symptom':
                symptom = dict_condition[i]
            elif i is 'provider':
                provider = dict_condition[i]
            elif i is 'tag':
                tag = dict_condition[i]

        sql = f"SELECT * FROM smart_mirror_system.remedy_method "

        if rm_id is not None:
            condition = f"WHERE rm_id = \'{rm_id}\'"

        elif symptom is not None and provider is not None and tag is not None:
            condition = f"WHERE symptom = \'{symptom}\' AND provider = \'{provider}\' AND tag = \'{tag}\'"

        elif symptom is not None and provider is not None and tag is None:
            condition = f"WHERE symptom = \'{symptom}\' AND provider = \'{provider}\'"

        elif symptom is not None and provider is None and tag is not None:
            condition = f"WHERE symptom = \'{symptom}\' AND tag = \'{tag}\'"

        elif symptom is not None and provider is None and tag is None:
            condition = f"WHERE symptom = \'{symptom}\'"

        elif symptom is None and provider is not None and tag is not None:
            condition = f"WHERE provider = \'{provider}\' AND tag = \'{tag}\'"

        elif symptom is None and provider is not None and tag is None:
            condition = f"WHERE provider = \'{provider}\'"

        elif symptom is None and provider is None and tag is not None:
            condition = f"WHERE tag = \'{tag}\'"

        elif symptom is None and provider is None and tag is None:
            condition = f""

        with self.conn.cursor() as cursor:
            try:
                print(sql+condition)
                cursor.execute(sql+condition)
                result = cursor.fetchall()

                if result is ():
                    return []
                else:
                    return result

            except Exception as e:
                print(e)
                return []

    def _db_retrieve_all_rm(self, symptom=None):

        if symptom:
            sql = f"SELECT * FROM remedy_method WHERE symptom = \'{symptom}\'"
        else:
            sql = f"SELECT * FROM remedy_method"

        with self.conn.cursor() as cursor:
            try:
                cursor.execute(sql)
                # print(sql)
                result = cursor.fetchall()

                if result is ():
                    return []
                else:
                    return result

            except Exception as e:
                print(e)
                return []

    def _db_deregister_remedy_method(self, rm_id):

        sql = f'DELETE FROM remedy_method WHERE rm_id = \'{rm_id}\''

        with self.conn.cursor() as cursor:
            try:
                cursor.execute(sql)
                print(f"Remedy method id: \'{rm_id}\' has been deleted.")
                return True

            except Exception as e:
                print(e)
                return False

    def _db_update_remedy_method(self, remedy_method_id, dict_info):

        cursor = self.conn.cursor()

        # To check dict_info has Type problem
        condition_string = sm.StringMaker().key_equation_value(dict_info)
        if condition_string is "":
            print("Invalid input in dict_info.")
            return False

        # To validate dict_info's keys
        for i in dict_info:
            if i not in ('rm_type', 'symptom', 'provider', 'url',
                         'maintain', 'improve', 'prevent', 'description', 'tag'):
                print(f"Invalid key type \'{i}\'. Please check your key again.")
                return False

        sql_check_method_valid = f'SELECT * FROM smart_mirror_system.remedy_method WHERE rm_id = \'{remedy_method_id}\''

        try:
            cursor.execute(sql_check_method_valid)
            result = cursor.fetchall()
            if result is ():
                print(f"There is no remedy method matching with id : \'{remedy_method_id}\'")
                return False

        except Exception as e:
            print(e)
            return False

        sql = f"UPDATE remedy_method SET {sm.StringMaker().key_equation_value(dict_info)}" \
            f" WHERE rm_id = \'{remedy_method_id}\'"

        try:
            cursor.execute(sql)
            print(f"Remedy method id: \'{remedy_method_id}\' has been updated")
            return True

        except Exception as e:
            print(e)
            return False


if __name__ == '__main__':
    rm = RemedyMethod()
    rm.register_remedy_method({"rm_type": 'test', "symptom": 'test', "provider": 'test', "url": 'test'
                               , "maintain": 1, "improve": 1, "description": 'testtest', "tag": 'testtest'})

    # rm.retrieve_all_rm()

    # print(rm.retrieve_rm_by_provider({"remedy_method_id": 12}))

    # rm.retrieve_rm_by_provider({"symptom":'baldness'})
    # print(rm.retrieve_rm_by_provider({"remedy_method_id": 108}))

    # rm.deregister_remedy_method(133)
    # rm.retrieve_remedy_method_by_id(114)
    # rm.retrieve_remedy_method_by_id()

    # rm.update_remedy_method(11, {"rm_type": 'aaa'})
