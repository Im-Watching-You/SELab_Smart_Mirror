# To manage Detected Age Table
# author  : Woo Chan Park
# version : 2019.06.21

import WC.dbconnector as db
import random
import pymysql.cursors
from datetime import datetime


class AgeAssessment:
    # Constructor
    def __init__(self,):
        self.conn = None

    # Create Age Assessment's record
    def register_detected_age(self, user_id=None, age=None, saved_path=None):

        # Check if all parameters are filled
        if user_id is not None and saved_path is not None and age is not None:

            # Integrity Check : age
            # condition : age > 0
            if age < 0:
                print("Invalid age.")
                return False

            # Integrity Check : saved_path
            # condition : how? photo taker will take care of it

            db.connect_to_db(self)
            with self.conn.cursor() as cursor:
                sql = 'INSERT INTO ageassessment(user_id, age, saved_path, recorded_date) VALUES(%s, %s, %s, %s)'
                try:
                    cursor.execute(sql, (user_id, age, saved_path, datetime.now()))
                    db.disconnect_from_db(self)
                    print("Assessed age Data has been Inserted.")
                    return True

                except pymysql.err.IntegrityError as i:
                    print(i)
                    db.disconnect_from_db(self)
                    return False

        # At least one parameter is missing
        else:
            print('Empty value detected')
            return False

    # Delete DetectedAge's record
    def deregister_detected_age(self, da_id):
        db.connect_to_db(self)

        with self.conn.cursor() as cursor:
            sql = 'DELETE FROM ageassessment where da_id = %s'

            try:
                cursor.execute(sql, da_id)
                db.disconnect_from_db(self)
                print("\'da_id : %s\' has deleted from DB" % da_id)
                return True

            except pymysql.err.IntegrityError as i:
                print(i)
                db.disconnect_from_db(self)
                return False

    def retrieve_by_user_id(self, user_id):
        db.connect_to_db(self)

        with self.conn.cursor() as cursor:
            sql = 'SELECT * FROM ageassessment WHERE user_id = %s'
            cursor.execute(sql, user_id)
            result = cursor.fetchall()

            if result:
                db.disconnect_from_db(self)
                return result
            else:
                db.disconnect_from_db(self)
                return None

    def retrieve_by_age(self, age):
        db.connect_to_db(self)

        if age < 0:
            print('Invalid data')
            return

        with self.conn.cursor() as cursor:
            sql = 'SELECT * FROM ageassessment WHERE age = %s'
            cursor.execute(sql, age)
            result = cursor.fetchall()

            if result:
                db.disconnect_from_db(self)
                return result
            else:
                db.disconnect_from_db(self)
                return None

    def retrieve_by_duration(self, start_date=None, end_date=None):

        # To check valid date data has been passed
        if start_date is not None and end_date is not None:

            # When start date exceeds end date
            if int((datetime.strptime(end_date, "%Y%m%d") - datetime.strptime(start_date, "%Y%m%d")).day) < 0:
                print("Invalid date")
                return False

            db.connect_to_db(self)
            with self.conn.cursor() as cursor:
                sql = 'SELECT * FROM ageassessment WHERE recorded_date BETWEEN %s AND %s'
                cursor.execute(sql, (start_date, end_date))
                result = cursor.fetchall()

                if result:
                    return result
                else:
                    return None

        # Only 'start_date' data has been passed : calculate from start_date to today
        elif start_date is not None and end_date is None:

            # When start date exceeds today
            if int((datetime.now() - datetime.strptime(start_date, "%Y%m%d")).day) < 0:
                print("Invalid date")
                return False

            db.connect_to_db(self)
            with self.conn.cursor() as cursor:
                sql = 'SELECT * FROM ageassessment WHERE recorded_date BETWEEN %s AND %s'
                cursor.execute(sql, (start_date, datetime.now()))
                result = cursor.fetchall()

                if result:
                    return result
                else:
                    return None
        else:
            print("Invalid date")
            return False


aa = AgeAssessment()
aa.register_detected_age(random, 24, '/root')
# aa.deregister_detected_age(24)

# aa.retrieve_by_duration('19991212')
# aa.retrieve_by_duration('19991212', '20190621')
# aa.retrieve_by_duration('29991212', '20190621')
