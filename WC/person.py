import WC.dbconnector as db
import WC.aaaapp as aa
import WC.stringmaker as sm
import pymysql
import WC.dummymaker as dm
import random

from datetime import datetime
from datetime import timedelta

import pandas as pd

class Person:

    def __init__(self):
        pass


class User(Person):

    def __init__(self):
        self.conn = None
        db.connect_to_db(self)
        self.aa = aa.AAAApp()

    def __del__(self):
        db.disconnect_from_db(self)

    def register_user(self, dict_info):
        """
        :param dict_info:
            key(1) password: String
            key(2) gender: String
            key(3) birth_date: Datetime
            key(4) first_name: String
            key(5) last_name: String
            key(6) phone_number: String
            key(7) email: String
            key(8) ap_id: Integer
        :return: boolean
        """
        return self._db_register_user(dict_info)

    def deregister_user(self, user_id):
        """
        :param user_id: Integer
        :return: boolean
        """
        return self._db_deregister_user(user_id)

    def retrieve_user(self, dict_condition=None):
        """
        :param dict_condition:
            key(1) first_name: String
            key(2) last_name: String
            key(3) user_id: Integer
            key(4) phone_number: String
            key(5) order_by: String
            key(6) start_joined: String with "YYYYMMDD" format
            key(7) end_joined: String with "YYYYMMDD" format

        :return: list of dictionaries
        """
        return self._db_retrieve_user(dict_condition)

    def retrieve_user_by_age_gender(self, dict_condition):
        """
        :param dict_condition:
            key(1) from: Integer
            key(2) to: Integer
            key(3) gender: String
        :return:
        """
        return self._db_retrieve_user_by_age_gender(dict_condition)

    def update_user_profile(self, user_id, dict_info):
        """
        :param user_id: Integer
        :param dict_info: Any attributes that you want to update
        :return:
        """
        return self._db_update_user(user_id, dict_info)

    def get_id_list(self):
        with self.conn.cursor() as cursor:
            sql = 'SELECT u_id, first_name, last_name FROM user'
            cursor.execute(sql)
            result = cursor.fetchall()

            if result is ():
                return []
            else:
                return result

    def make_dummy_user(self, count=100):

        for i in range(count):
            name = dm.dummy_name()
            self.register_user('test', 'male', dm.dummy_date(), name[0], name[1], dm.dummy_phone_number(),
                               dm.dummy_email())

        for i in range(count):
            name = dm.dummy_name('female')
            self.register_user('test', 'female', dm.dummy_date(), name[0], name[1], dm.dummy_phone_number(),
                               dm.dummy_email())

    def truncate_user(self):
        return self._db_truncate_user()

    def _db_register_user(self, dict_profile):

        cursor = self.conn.cursor()

        try:
            password = dict_profile['password']
            gender = dict_profile['gender']
            birth_date = dict_profile['birth_date']
            first_name = dict_profile['first_name']
            last_name = dict_profile['last_name']
            phone_number = dict_profile['phone_number']
            email = dict_profile['email']
            ap_id = dict_profile['ap_id']

        except KeyError as e:
            print("ERROR : WC.person._db_register_user")
            print(f"Please check your key name again.")
            print(str(e) + ' is missing.')
            return False

        self.age = int((datetime.now() - datetime.strptime(list(dict_profile.values())[2], "%Y%m%d")).days / 365)

        sql = f'INSERT INTO user(password ,gender, birth_date, first_name, last_name, ' \
              f'phone_number, email, ap_id, joined_date, age) ' \
              f'VALUES(\'{password}\',\'{gender}\',\'{birth_date}\',\'{first_name}\', ' \
              f'\'{last_name}\',\'{phone_number}\',\'{email}\',\'{ap_id}\',\'{datetime.now()}\',\'{self.age}\')'
        try:
            cursor.execute(sql)
            print("User's Data has been Inserted.")
            return True

        except Exception as e:
            print(e)
            return False

    def _db_deregister_user(self, user_id):
        """
        :param user_id:
        :return: boolean
        """

        sql = f'DELETE FROM user WHERE id = \'{user_id}\''

        with self.conn.cursor() as cursor:
            try:
                cursor.execute(sql)
                print(f'User id : \'{user_id}\' has been deleted.')
                return True
            except Exception as e:
                print(e)
                return False

    def _db_retrieve_user(self, dict_condition=None):

        sql = 'SELECT * FROM smart_mirror_system.user '

        condition = ""

        order = ""

        first_name, last_name, user_id, phone_number, order_by = None, None, None, None, ""

        start_joined, end_joined = '20000101', (datetime.now()+timedelta(days=1)).strftime("%Y%m%d")

        if dict_condition is not None:

            for i in list(dict_condition):
                if i is 'first_name':
                    first_name = dict_condition[i]
                elif i is 'last_name':
                    last_name = dict_condition[i]
                elif i is 'user_id':
                    user_id = dict_condition[i]
                elif i is 'phone_number':
                    phone_number = dict_condition[i]
                elif i is 'start_joined':
                    start_joined = dict_condition[i]
                elif i is 'end_joined':
                    end_joined = dict_condition[i]
                elif i is 'order_by':
                    order_by = dict_condition[i]
                else:
                    print("ERROR : WC.person._db_retrieve_user")
                    print(f'Invalid key name \'{i}\'. Please Check your key name again.')
                    return False

        # print(phone_number)

        if user_id is not None:
            condition = f"WHERE id = \'{user_id}\'"

        elif phone_number is not None:
            condition = f'WHERE phone_number = \'{phone_number}\''

        elif first_name and last_name:
            condition = f'WHERE first_name = \'{first_name}\' AND last_name = \'{last_name}\' AND joined_date ' \
                f'>= \'{start_joined}\' AND joined_date < \'{end_joined}\''

        elif first_name is not None and last_name is None:
            condition = f'WHERE first_name = \'{first_name}\' AND joined_date ' \
                f'>= \'{start_joined}\' AND joined_date < \'{end_joined}\''

        elif first_name is None and last_name is not None:
            condition = f'WHERE last_name = \'{last_name}\' AND joined_date ' \
                f'>= \'{start_joined}\' AND joined_date < \'{end_joined}\''

        else:
            condition = f'WHERE joined_date >= \'{start_joined}\' ' \
                        f'AND joined_date <= \'{end_joined}\''

        if order_by is not '':
            if order_by is 'rand':
                order = ' ORDER BY rand()'
            elif order_by is 'male_first':
                order = ' ORDER BY gender DESC'
            elif order_by is 'female_first':
                order = ' ORDER BY gender ASC'
            elif order_by is 'birth_asc':
                order = ' ORDER BY birth_date ASC'  # older first
            elif order_by is 'birth_desc':
                order = ' ORDER BY birth_date DESC'  # younger first
            elif order_by is 'joined_date_asc':
                order = ' ORDER BY joined_date ASC'
            elif order_by is 'joined_date_desc':
                order = ' ORDER BY joined_date DESC'
            else:
                print("Invalid ordering option")
                return []

        condition = condition + order

        with self.conn.cursor() as cursor:
            try:
                # print(sql+condition)
                cursor.execute(sql+condition)
                result = cursor.fetchall()

                if result is ():
                    return []
                else:
                    return result

            except Exception as e:
                print(e)
                print("__db_retrieve_user: error")
                return []

    def _db_retrieve_user_by_age_gender(self, dict_condition):

        age_from, age_to, gender = 0, 99, None

        for i in list(dict_condition):
            if i is 'from':
                age_from = dict_condition[i]
            elif i is 'to':
                age_to = dict_condition[i]
            elif i is 'gender':
                gender = dict_condition[i]

        if gender:
                sql = f"SELECT * FROM smart_mirror_system.user WHERE age >= \'{age_from}\' AND age <= \'{age_to}\' " \
                      f"AND gender = \'{gender}\'"
        else:
                sql = f"SELECT * FROM smart_mirror_system.user WHERE age >= \'{age_from}\' AND age <= \'{age_to}\' "

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
                print("__db_retrieve_user_by_age: error")
                return []

    def _db_update_user(self, user_id, dict_info):

        condition_string = sm.StringMaker().key_equation_value(dict_info)

        if condition_string is "":
            return False

        sql = f"UPDATE user SET {condition_string} " \
            f"WHERE id = {user_id}"

        with self.conn.cursor() as cursor:
            try:
                is_updated = cursor.execute(sql)
                if is_updated:
                    print(f"User id : {user_id} has been updated.")
                else:
                    print(f"There is no matching id or Input data has already been applied.")
                return True

            except Exception as e:
                print(e)
                return False

    def _db_truncate_user(self):
        option = input('<STAFF ONLY> This method will delete entire rows of the table. Are you Sure? (Y/N)')

        if option.lower() == 'y':
            with self.conn.cursor() as cursor:
                try:
                    sql = "TRUNCATE TABLE user; ALTER TABLE user auto_increment = 1;"
                    cursor.execute(sql)
                    return True

                except Exception as e:
                    print(e)
                    return False
        else:
            print('Truncate Canceled.')
            return False


class Staff(Person):

    def __init__(self):
        self.conn = None
        db.connect_to_db(self)

    def __del__(self):
        db.disconnect_from_db(self)

    def register_staff(self, dict_info):
        """
        :param dict_info:
            key(1) password: String
            key(2) gender: String
            key(3) birth_date: Datetime
            key(4) first_name: String
            key(5) last_name: String
            key(6) phone_number: String
            key(7) email: String
        :return: boolean
        """
        return self._db_register_staff(dict_info)

    def deregister_staff(self, staff_id):
        """
        :param staff_id: Integer
        :return: boolean
        """
        return self._db_deregister_staff(staff_id)

    def retrieve_staff(self, dict_condition=None):
        """
        :param dict_condition:
            key(1) first_name: String
            key(2) last_name: String
            key(3) user_id: Integer
            key(4) phone_number: String
            key(5) order_by: String
            key(6) start_joined: String with "YYYYMMDD" format
            key(7) end_joined: String with "YYYYMMDD" format

        :return: list of dictionaries
        """
        return self._db_retrieve_staff(dict_condition)

    def update_staff_profile(self, staff_id, dict_info):
        """
        :param staff_id: Integer
        :param dict_info: Any attributes that you want to update
        :return:
        """
        return self._db_update_staff(staff_id, dict_info)

    def make_dummy_staff(self, count=50):

        for i in range(count):
            name = dm.dummy_name()
            self.register_staff('test', 'male', dm.dummy_date(), name[0], name[1], dm.dummy_phone_number(),
                               dm.dummy_email(), random.randrange(1, 5))

        for i in range(count):
            name = dm.dummy_name('female')
            self.register_staff('test', 'female', dm.dummy_date(), name[0], name[1], dm.dummy_phone_number(),
                               dm.dummy_email(), random.randrange(1, 5))

    def truncate_staff(self):
        return self._db_truncate_staff()

    def _db_register_staff(self, dict_profile):

        try:
            password = dict_profile['password']
            gender = dict_profile['gender']
            birth_date = dict_profile['birth_date']
            first_name = dict_profile['first_name']
            last_name = dict_profile['last_name']
            phone_number = dict_profile['phone_number']
            email = dict_profile['email']
            tier = dict_profile['tier']

        except KeyError as e:
            print("ERROR : WC.person._db_register_staff")
            print("Please check your key name again.")
            print(e)
            return False

        with self.conn.cursor() as cursor:

            self.age = int((datetime.now() - datetime.strptime(list(dict_profile.values())[2], "%Y%m%d")).days / 365)

            sql = f'INSERT INTO staff(password ,gender, birth_date, first_name, last_name, ' \
                  f'phone_number, email, tier, joined_date, age) ' \
                  f'VALUES(\'{password}\',\'{gender}\',\'{birth_date}\',\'{first_name}\', ' \
                  f'\'{last_name}\',\'{phone_number}\',\'{email}\',\'{tier}\',\'{datetime.now()}\',\'{self.age}\')'
            try:
                # print(self.birth_date)  # for debugging

                cursor.execute(sql)

                print("Staff's Data has been Inserted.")
                return True

            except pymysql.err.IntegrityError as i:
                print(i)
                return False

            except ValueError:
                print("UNSOLVED VALUE ERROR")  # for debugging
                return False

    def _db_deregister_staff(self, staff_id):

        sql = f'DELETE FROM staff WHERE id = \'{staff_id}\''

        with self.conn.cursor() as cursor:
            try:
                cursor.execute(sql)
                print(f'Staff id : \'{staff_id}\' has been deleted.')
                return True

            except Exception as e:
                print(e)
                return False

    def _db_retrieve_staff(self, dict_condition):

        sql = 'SELECT * FROM staff '
        condition = ""
        order = ""

        first_name, last_name, user_id, phone_number, order_by = None, None, None, None, ""
        start_joined, end_joined = '20000101', (datetime.now()+timedelta(days=1)).strftime("%Y%m%d")

        if dict_condition is not None:
            for i in list(dict_condition):
                if i is 'first_name':
                    first_name = dict_condition[i]
                elif i is 'last_name':
                    last_name = dict_condition[i]
                elif i is 'user_id':
                    user_id = dict_condition[i]
                elif i is 'phone_number':
                    phone_number = dict_condition[i]
                elif i is 'start_joined':
                    start_joined = dict_condition[i]
                elif i is 'end_joined':
                    end_joined = dict_condition[i]
                elif i is 'order_by':
                    order_by = dict_condition[i]
                else:
                    print("ERROR : WC.person._db_retrieve_staff")
                    print(f'Invalid key name \'{i}\'. Please Check your key name again.')
                    return False

        if user_id is not None:
            condition = f"WHERE id = \'{user_id}\'"

        elif phone_number is not None:
            condition = f'WHERE phone_number = \'{phone_number}\''

        elif first_name and last_name:
            condition = f'WHERE first_name = \'{first_name}\' AND last_name = \'{last_name}\' AND joined_date ' \
                f'>= \'{start_joined}\' AND joined_date < \'{end_joined}\''

        elif first_name is not None and last_name is None:
            condition = f'WHERE first_name = \'{first_name}\' AND joined_date ' \
                f'>= \'{start_joined}\' AND joined_date < \'{end_joined}\''

        elif first_name is None and last_name is not None:
            condition = f'WHERE last_name = \'{last_name}\' AND joined_date ' \
                f'>= \'{start_joined}\' AND joined_date < \'{end_joined}\''

        else:
            condition = f'WHERE joined_date >= \'{start_joined}\' ' \
                        f'AND joined_date < \'{end_joined}\''

        if order_by is not '':
            if order_by is 'rand':
                order = ' ORDER BY rand()'
            elif order_by is 'male_first':
                order = ' ORDER BY gender DESC'
            elif order_by is 'female_first':
                order = ' ORDER BY gender ASC'
            elif order_by is 'birth_asc':
                order = ' ORDER BY birth_date ASC'  # older first
            elif order_by is 'birth_desc':
                order = ' ORDER BY birth_date DESC'  # younger first
            elif order_by is 'joined_date_asc':
                order = ' ORDER BY joined_date ASC'
            elif order_by is 'joined_date_desc':
                order = ' ORDER BY joined_date DESC'
            else:
                print("Invalid ordering option")
                return []

        condition = condition + order

        with self.conn.cursor() as cursor:
            try:
                # print(sql+condition)
                cursor.execute(sql+condition)
                result = cursor.fetchall()
                if result is ():
                    return []
                else:
                    return result

            except Exception as e:
                print(e)
                print("__db_retrieve_user: error")
                return []

    def _db_update_staff(self, staff_id, dict_info):

        condition_string = sm.StringMaker().key_equation_value(dict_info)

        if condition_string is "":
            return False

        sql = f"UPDATE staff SET {condition_string} " \
            f"WHERE id = {staff_id}"

        with self.conn.cursor() as cursor:
            try:
                is_updated = cursor.execute(sql)
                if is_updated:
                    print(f"Staff id : {staff_id} has been updated.")
                else:
                    print(f"There is no matching id or Input data has already been applied.")
                return True

            except Exception as e:
                print(e)
                return False

    def _db_truncate_staff(self):
        option = input('<STAFF ONLY> This method will delete entire rows of the table. Are you Sure? (Y/N)')

        if option.lower() == 'y':
            with self.conn.cursor() as cursor:
                try:
                    sql = "TRUNCATE TABLE smart_mirror_sytem.staff; ALTER TABLE user auto_increment = 1;"
                    cursor.execute(sql)
                    return True

                except Exception as e:
                    print(e)
                    return False
        else:
            print('Truncate Canceled.')
            return False


if __name__ == '__main__':
    user = User()
    staff = Staff()

    # user.register_user({"pasword": 'test123', "gender": 'male', "birth_date": '19940915',
    #                     "first_name": 'stephen', 'last_name': 'Kim', 'phone_number': '01026519876',
    #                     'email': 'test12933333@test.com', 'ap_id': 1})
    #
    # user.deregister_user(292)

    print(user.retrieve_user({'phone_number': '01023234545'}))
    # user.retrieve_user({"order_by": 'rand'})
    # user.retrieve_user({"first_name": 'John', 'last_name': 'Doe'})
    # print(user.retrieve_user({"last_name": 'Kim'}))

    # print(type(user.retrieve_user({"start_joined": '20190624'})))
    # print(user.retrieve_user({"start_joined": '20190624'}))
    #
    # user.retrieve_user_by_age_gender({'from': 12, 'to': 30, 'gender': 'male'})
    # user.retrieve_user_by_age_gender({'from': 12, 'to': 30, 'gender': 'female'})
    # user.retrieve_user_by_age_gender({'from': 12, 'to': 30})
    #
    # user.update_user_profile(user_id=1, dict_info={'password': '1234','gender': 'female'})
    #
    # staff.register_staff({"password": 'test123', "gender": 'male', "birth_date": '19940915',
    #                     "first_name": 'Woochan', 'last_name': 'Park', 'phone_number': '01051019876',
    #                     'email': 'tes2t1229875@test.com'})
    #
    # staff.deregister_staff(21)
    # staff.retrieve_staff({"first_name": 'john'})

    # staff.retrieve_staff()
    # staff.retrieve_staff({"first_name": 'Joe'})
    # staff.update_staff_profile(staff_id=2, dict_info={'password': '1234','gender': 'female'})

    # age_input = {'from': 100}
    # print(pd.DataFrame(user.retrieve_user_by_age_gender(age_input)))
    # print(type(user.retrieve_user_by_age_gender(age_input)))