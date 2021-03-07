#################################################################
# !!!!!!!!!!!!!!  We don't use this code anymore  !!!!!!!!!!!!!!#
#################################################################

# Now we use 'remedymethod.py'

import pymysql.cursors
import numpy as np
import pandas as pd
from datetime import datetime
from IPython.display import display

class Recommendation:

    # Constructor
    # Insert record into DB as soon as instance is created
    # Data Integrity concerned
    def __init__(self, rm_type=None, symptom=None, provider=None, url=None, effectiveness=None, description=None):
        # If all parameters are valid, then insert a record into DB
        if rm_type is not None and symptom is not None and provider is not None and url is not None and effectiveness is not None and description is not None:
            if effectiveness is 'maintain':
                self.connect_to_db()
                self.info = dict(rm_type=rm_type, symptom=symptom, provider=provider, url=url, maintain=1, improve=0, prevent=0, description=description, edit_date=datetime.now())
                self.register_recommendation()
                self.disconnect_from_db()
                del self

            elif effectiveness is 'improve':
                self.connect_to_db()
                self.info = dict(rm_type=rm_type, symptom=symptom, provider=provider, url=url, maintain=0, improve=1, prevent=0, description=description, edit_date=datetime.now())
                self.register_recommendation()
                self.disconnect_from_db()
                del self

            elif effectiveness is 'prevent':
                self.connect_to_db()
                self.info = dict(rm_type=rm_type, symptom=symptom, provider=provider, url=url, maintain=0, improve=0, prevent=1, description=description, edit_date=datetime.now())
                self.register_recommendation()
                self.disconnect_from_db()
                del self

            else:
                print("Invalid effectiveness.")
                del self

        # For CRUD operation
        # All parameter is None
        elif rm_type is None and symptom is None and provider is None and url is None and effectiveness is None and description is None:
            self.connect_to_db()

        # At least one parameter is missing
        else:
            print('Empty value detected')
            del self;

    # Configure DB setting
    def connect_to_db(self):
        try:
            self.conn = pymysql.connect(host='203.253.23.27', user='root', password='root', db='Smart_Mirror_System', charset= 'utf8', autocommit=True, cursorclass=pymysql.cursors.DictCursor)
            return True
        except Exception:
            print('Connection Error')
            return False

    def disconnect_from_db(self):
        self.conn.close()

    # Create recommendation's record
    # Data Integrity concerned
    def register_recommendation(self):
        rm_info = self.info
        with self.conn.cursor() as cursor:
            sql = 'INSERT INTO recommendation(rm_type, symptom, provider, url, maintain, improve, prevent, description, edit_date) VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s)'
            try:
                cursor.execute(sql, (rm_info['rm_type'], rm_info['symptom'], rm_info['provider'], rm_info['url'], rm_info['maintain'], rm_info['improve'], rm_info['prevent'], rm_info['description'], rm_info['edit_date']))
                return True
            except pymysql.err.IntegrityError as i:
                print(i)
                return False

    def deregister_recommendation(self, rm_id):
        with self.conn.cursor() as cursor:
            sql = 'DELETE FROM recommendation where rm_id = %s'
            try:
                cursor.execute(sql, rm_id)
                return True
            except pymysql.err.IntegrityError as i:
                print(i)
                return False


# rm = Recommendation()
# abc = Recommendation(1,1,1,1,'maintain',3)
# rm.deregister_recommendation(2);
