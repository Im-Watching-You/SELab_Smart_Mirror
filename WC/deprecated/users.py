#################################################################
# !!!!!!!!!!!!!!  We don't use this code anymore  !!!!!!!!!!!!!!#
#################################################################

# Now we use 'usermanager.py'

# To manage User's information
# author  : Woo Chan Park
# version : 2019.06.18

import dbconnector as db
import framemaker as fm
import pymysql
import numpy as np
import pandas as pd
from datetime import datetime
from IPython.display import display


class Users:

    # Constructor
    # Insert record into DB as soon as instance is created
    # Data Integrity concerned
    def __init__(self, gender=None, birth_date=None, first_name=None, last_name=None, phone_number=None, email=None):
        # If all parameters are valid, then insert a record into DB
        self.conn = None
        if gender is not None and birth_date is not None and first_name is not None and last_name is not None and phone_number is not None and email is not None:
            db.connect_to_db(self)
            self.info = dict(gender=gender, birth_date=birth_date, first_name=first_name, last_name=last_name, phone_number=phone_number, email=email, joined_date=datetime.now())
            self.register_id()
            db.disconnect_from_db(self)
            # self.disconnect_from_db()
            del self

        # For CRUD usage
        elif gender is None and birth_date is None and first_name is None and last_name is None and phone_number is None and email is None:
            db.connect_to_db(self)

        # At least one parameter is missing
        else:
            print('Empty value detected')
            del self

    # Create User's record
    # Data Integrity concerned
    def register_user(self):
        user_info = self.info
        with self.conn.cursor() as cursor:
            sql = 'INSERT INTO users(gender, birth_date, first_name, last_name, phone_number, email, joined_date) VALUES(%s, %s, %s, %s, %s, %s, %s)'
            try:
                cursor.execute(sql, (user_info['gender'], user_info['birth_date'], user_info['first_name'], user_info['last_name'], user_info['phone_number'], user_info['email'], user_info['joined_date']))
                return True
            except pymysql.err.IntegrityError as i:
                return False

    # Delete User's record
    # Need to handle '0 row affected' when duplicate deletion of record
    def deregister_user(self, phone_number):
        with self.conn.cursor() as cursor:
            sql = 'DELETE FROM users WHERE phone_number = %s'
            try:
                cursor.execute(sql, (phone_number))
                # print('done')
                return True
            except Exception as e:
                print(e)
                return False

    # Update ID information
    def update_user(self, phone_number, ):
        with self.conn.cursor() as cursor:
            sql = 'UPDATE users set ~ where phone_number = %s'
            try:
                cursor.execute(sql, (phone_number))
                return True
            except Exception as e:
                print(e)
                return False

    # Retrieve User's information by User's id
    def retrieve_by_id(self, user_id):
        with self.conn.cursor() as cursor:
            sql = 'SELECT * FROM users WHERE user_id = 120'
            cursor.execute(sql)
            result = cursor.fetchone()

            if result:
                return fm.UsersFrame.make_users_frame(result)
            else:
                return None

    # Get list of all users' id
    def get_id_list(self):
        with self.conn.cursor() as cursor:
            sql = 'SELECT user_id FROM users'
            cursor.execute(sql)
            result = cursor.fetchall()

            if result:
                return result
            else:
                return None
