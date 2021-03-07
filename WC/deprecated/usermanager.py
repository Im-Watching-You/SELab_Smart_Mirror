# To manage User's information
# author  : Woo Chan Park
# version : 2019.06.22

import WC.dbconnector as db
import pymysql
import re
import WC.dummymaker as dm
from datetime import datetime


class UserManager:
    # Constructor
    def __init__(self):
        self.conn = None

    # Create User's record
    def register_user(self, password=None, gender=None, birth_date=None, first_name=None, last_name=None,
                      phone_number=None, email=None):

        # If all parameters are filled, then check the validity of parameters
        if password is not None and gender is not None and birth_date is not None and first_name is not \
                None and last_name is not None and phone_number is not None and email is not None:

            # Integrity Check : password
            # condition : max(len) = 16
            if len(password) > 16:
                print('Maximum length of Password should be under 16 number of characters.')
                return False

            # Integrity Check : gender
            # condition : only male or female
            if gender.lower() != 'male' and gender.lower() != 'female':
                print(gender.lower() is 'male')
                print('Invalid Gender. Please insert \'male\'or \'female\' regardless of Case.')
                return False

            # Integrity Check : birth_date
            # condition : '19940915' format
            # When Birth Date exceeds today : The man who came from the future
            try:
                if (datetime.now() - datetime.strptime(birth_date, "%Y%m%d")).days < 0:
                    # https://brownbears.tistory.com/18
                    print('Invalid Birth Date.')
                    return False
            except ValueError:
                    print()
            # Integrity Check : phone_number
            # condition : '01051319925' format
            if phone_number[0:3] != '010' or len(phone_number) is not 11:
                print('Invalid Phone Number')
                return False

            # Integrity Check : email
            # condition : 'ABC@BBB.com' format
            check_email = re.compile('^[a-zA-Z0-9+-_.]+@[a-zA-Z0-9-]+.[a-zA-Z0-9-.]+$')
            # https://dojang.io/mod/page/view.php?id=2439
            if check_email.match(email) is None:
                print('Invalid Email Address')
                return False

            # Insert Data into DB
            db.connect_to_db(self)
            with self.conn.cursor() as cursor:
                sql = 'INSERT INTO user(password ,gender, age, birth_date, first_name, last_name, ' \
                      'phone_number, email, joined_date) VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s)'
                try:
                    print(birth_date)       # for debugging
                    age = int((datetime.now() - datetime.strptime(birth_date, "%Y%m%d")).days / 365)
                    cursor.execute(sql, (password, gender, age, birth_date, first_name, last_name, phone_number,
                                         email, datetime.now()))
                    print("User's Data has been Inserted.")
                    db.disconnect_from_db(self)
                    return True

                except pymysql.err.IntegrityError as i:
                    print("INTEGRITY CONSTRAINTS VIOLATION")
                    db.disconnect_from_db(self)
                    return False

                # 19900229 : 윤년이 아닌데 29일까지 생성되었다.
                # 19930229 : 윤년이 아닌데 29일까지 생성되었다.
                except ValueError:
                    print("UNSOLVED VALUE ERROR")       # for debugging

        # At least one parameter is missing
        else:
            print('Empty value detected')
            return False

    # Delete User's record
    # Need to handle '0 row affected' when duplicate deletion of record
    def deregister_user(self, phone_number):
        db.connect_to_db(self)
        with self.conn.cursor() as cursor:
            sql = 'DELETE FROM user WHERE phone_number = %s'
            try:
                cursor.execute(sql, phone_number)
                db.disconnect_from_db(self)
                return True
            except Exception as e:
                print(e)
                db.disconnect_from_db(self)
                return False

    # Get Whole Table by passed sorting option and duration
    def retrieve_whole_table(self, order_by=None, start_date=None, end_date=None):

        sql = 'SELECT * FROM user '

        # 전체 기간 탐색
        if start_date is None and end_date is None:
            sql_date = ''

        # start_date 부터 현재 까지
        elif start_date is not None and end_date is None:
            sql_date = 'WHERE joined_date BETWEEN ' + start_date + ' AND ' + str(datetime.now().strftime("%Y%m%d"))

        # start_date 부터 end_date 까지
        elif start_date is not None and end_date is not None:
            sql_date = 'WHERE joined_date BETWEEN ' + start_date + ' AND ' + end_date

        else:
            print('Invalid date')
            return None

        if order_by is None:
            sql_order = ''
        elif order_by is 'rand':
            sql_order = ' ORDER BY rand()'
        elif order_by is 'male_first':
            sql_order = ' ORDER BY gender ASC'
        elif order_by is 'female_first':
            sql_order = ' ORDER BY gender DESC'
        elif order_by is 'birth_asc':
            sql_order = ' ORDER BY birth_date ASC'     # older first
        elif order_by is 'birth_desc':
            sql_order = ' ORDER BY birth_date DESC'    # younger first
        elif order_by is 'joined_date_asc':
            sql_order = ' ORDER BY joined_date ASC'
        elif order_by is 'joined_date_desc':
            sql_order = ' ORDER BY joined_date DESC'

        else:
            print("Invalid ordering option")
            return None

        db.connect_to_db(self)
        with self.conn.cursor() as cursor:
            try:
                cursor.execute(sql + sql_date + sql_order)
                print(sql + sql_date + sql_order)
                result = cursor.fetchall()
                db.disconnect_from_db(self)
                return result

            except Exception as e:
                print(e)
                db.disconnect_from_db(self)
                return None

    # Update ID information
    def update_user_info(self, u_id, password=None, phone_number=None, email=None):

        # Integrity Check : password
        # condition : max(len) = 16
        if len(password) > 16:
            print('Maximum length of Password should be under 16 number of characters.')
            return False

        # Integrity Check : phone_number
        # condition : '01012345678' format
        if phone_number[0:3] != '010' or len(phone_number) is not 11:
            print('Invalid Phone Number')
            return False

        # Integrity Check : email
        # condition : 'ABC@BBB.com' format
        check_email = re.compile('^[a-zA-Z0-9+-_.]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$')
        # https://dojang.io/mod/page/view.php?id=2439

        if check_email.match(email) is None:
            print('Invalid Email Address')
            return False

        sql_top = 'UPDATE user SET '
        sql_bot = ' WHERE u_id = ' + str(u_id)

        if password is not None and phone_number is not None and email is not None:
            sql_mid = 'password = ' + password + ', phone_number = ' + phone_number + ', email = \'' + email + '\''
        elif password is not None and phone_number is not None and email is None:
            sql_mid = 'password = ' + password + ', phone_number = ' + phone_number
        elif password is not None and email is not None:
            sql_mid = 'password = ' + password + ', email = ' + email
        else:
            print('Parameters are missing')
            return False

        db.connect_to_db(self)
        with self.conn.cursor() as cursor:
            try:
                # print(sql_top+sql_mid+sql_bot)
                cursor.execute(sql_top + sql_mid + sql_bot)
                db.disconnect_from_db(self)
                return True
            except Exception as e:
                print(e)
                db.disconnect_from_db(self)
                return False

    # Retrieve User's information by User's id
    def retrieve_by_id(self, u_id):
        db.connect_to_db(self)
        with self.conn.cursor() as cursor:
            sql = 'SELECT * FROM user WHERE u_id = %s'
            cursor.execute(sql, u_id)
            result = cursor.fetchall()
            if result:
                db.disconnect_from_db(self)
                return result
            else:
                db.disconnect_from_db(self)
                return None

    # Get list of all users' id , first_name, last_name
    def get_id_list(self):
        db.connect_to_db(self)
        with self.conn.cursor() as cursor:
            sql = 'SELECT u_id, first_name, last_name FROM user'
            cursor.execute(sql)
            result = cursor.fetchall()

            if result:
                db.disconnect_from_db(self)
                return result
            else:
                db.disconnect_from_db(self)
                return None

    # make dummy_data of users
    def make_dummy_user(self, amount=100):
        for i in range(int(amount/2)):
            name = dm.dummy_name()
            self.register_user('test', 'male', dm.dummy_date(), name[0], name[1], dm.dummy_phone_number(), dm.dummy_email())

        for i in range(int(amount/2)):
            name = dm.dummy_name('female')
            self.register_user('test', 'female', dm.dummy_date(), name[0], name[1], dm.dummy_phone_number(), dm.dummy_email())

# a = UserManager()
# a.make_dummy_data()
# print(a.get_id_list())
# print(a.retrieve_whole_table(start_date='19990101', end_date='20190621'))

# a.update_user_info(24,'01032323232','@gmail.com')              #invalid
# a.update_user_info(24, '01032323232', 'wch940@gmail.com')


