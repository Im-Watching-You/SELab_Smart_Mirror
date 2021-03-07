import sys
import pymysql
import json
from pymysql.err import OperationalError
import datetime
import time
# from WC.remedymethod import RemedyMethod

class DataManager():

    def __init__(self):
        host = "203.253.23.22"
        port = 3306
        user = "root"
        password = "root"
        db_name = "smart_mirror_system"
        try:
            self.db_connector=pymysql.connect(host=host, port=port, user=user, passwd=password, db=db_name)
            self.isConnected = True
        except OperationalError as e:
            print(e)
            self.isConnected = False

    def _parse_result(self, param_list, result_str_list, underscore=False):
        """
        :param param_list: // List of Parameters
        :param result_str_list: // Data from DB
        :return:
        """
        result = {}
        for i, param in enumerate(param_list):
            # renamed = param
            # if not underscore:
            #     underscoreAmount = param.count('_')
            #     j = 0
            #     while j < underscoreAmount:
            #         loc = param.find("_")
            #         if loc >= 0:
            #             renamed = param[:loc]+param[loc+1].upper()
            #             if len(param) > loc+1:
            #                 renamed = renamed+param[loc+2:]
            #             param = renamed
            #         j = j+1
            result[param] = result_str_list[i]
        return result

    def _parse_aws_result(self, result_list, underscore=False):
        result = {}
        for keyName in result_list:
            renamed = keyName
            if not underscore:
                for c in keyName:
                    if c.isupper():
                        j = renamed.index(c)
                        renamed = renamed[:j]+'_'+renamed[j].lower()+renamed[j+1:]
                result[renamed] = result_list[keyName]
        return result

    def add_user(self, user):
        if_inserted = False
        if type(user) is str:
            user = json.loads(user)
        user_id = user["user_id"]
        password = user["password"]
        firstname = user["firstname"]
        lastname = user["lastname"]
        birthday = user["birthday"]
        gender = user["gender"]
        age = user["age"]
        email = user["email"]
        phone = user["phone"]
        with self.db_connector.cursor() as cursor:
            try:
                db_query = "INSERT INTO user (user_id, password, firstname, lastname, birthday, gender, age, email, phone) values (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
                cursor.execute(db_query, (user_id, password, firstname, lastname, birthday, gender, age, email, phone))
                self.db_connector.commit()
                row_num = cursor.rowcount
                if row_num > -1:
                    if_inserted = True
            except Exception as e:
                print("Exception: ", e)
        return if_inserted

    def get_user(self, user_id, password):
        db_query = "SELECT * FROM user WHERE user_id=%s AND password=%s"
        user = {}
        with self.db_connector.cursor() as cursor:
            try:
                cursor.execute(db_query, (user_id, password))
                self.db_connector.commit()
                param_list = ["user_id", "password", "firstname", "lastname", "birthday", "gender", "age", "email", "phone"]
                for row in cursor:
                    user = self._parse_result(param_list, row, underscore=False)
                user["birthday"] = user["birthday"].strftime('%Y-%m-%d')
            except Exception as e:
                print("Exception: ", e)
        return user

    def add_rm(self, remedy_method):
        if_inserted = False
        if type(remedy_method) is str:
            remedy_method = json.loads(remedy_method)
        rm_type = remedy_method["rm_type"]
        symptom = remedy_method["symptom"]
        provider = remedy_method["provider"]
        url = remedy_method["url"]
        effc_degree = remedy_method["effc_degree"]
        maintain = remedy_method["maintain"]
        improve = remedy_method["improve"]
        prevent = remedy_method["prevent"]
        description = remedy_method["description"]
        edit_date = remedy_method["edit_date"]
        with self.db_connector.cursor() as cursor:
            try:
                db_query = "INSERT INTO remedy_method (rm_type, symptom, provider, url, effc_degree, maintain, improve, prevent, description, edit_date) values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                cursor.execute(db_query, (rm_type, symptom, provider, url, effc_degree, maintain, improve, prevent, description, edit_date))
                self.db_connector.commit()
                row_num = cursor.rowcount
                if row_num > -1:
                    if_inserted = True
            except Exception as e:
                print("Exception: ", e)
        return if_inserted

    def get_rm(self, rm_id):
        dbQuery = "SELECT * FROM remedy_method WHERE rm_id=%s"
        remedyMethod = {}
        with self.db_connector.cursor() as cursor:
            try:
                cursor.execute(dbQuery, (rm_id))
                self.db_connector.commit()
                param_list = ["rm_id", "rm_type", "symptom", "provider", "url", "maintain", "improve", "prevent", "description", "edit_date"]
                for row in cursor:
                    remedyMethod = self._parse_result(param_list, row, underscore=False)
                # user["birthday"] = user["birthday"].strftime('%Y-%m-%d')
            except Exception as e:
                print("Exception: ", e)
        return remedyMethod

    def get_all_rm(self, symptom):
        dbQuery = "SELECT * FROM remedy_method WHERE symptom=%s"
        rm = {}
        rmList = []
        with self.db_connector.cursor() as cursor:
            try:
                cursor.execute(dbQuery, (symptom))
                self.db_connector.commit()
                param_list = ["rm_id", "rm_type", "symptom", "provider", "url", "maintain", "improve", "prevent", "description", "edit_date", "tag"]
                for row in cursor:
                    rm = self._parse_result(param_list, row, underscore=False)
                    rmList.append(rm)
                # user["birthday"] = user["birthday"].strftime('%Y-%m-%d')
            except Exception as e:
                print("Exception: ", e)
        return rmList

    def get_recoms(self, user_id, rm_id=None, recom_rating=None):
        recomList = []
        with self.db_connector.cursor() as cursor:
            try:
                if rm_id is not None:
                    if recom_rating is not None:
                        dbQuery = "SELECT * FROM recommendation WHERE user_id=%s AND rm_id=%s AND recom_rating>=%s"
                        cursor.execute(dbQuery, (user_id, rm_id, recom_rating))
                    else:
                        dbQuery = "SELECT * FROM recommendation WHERE user_id=%s AND rm_id=%s"
                        cursor.execute(dbQuery, (user_id, rm_id))
                else:
                    if recom_rating is not None:
                        dbQuery = "SELECT * FROM recommendation WHERE user_id=%s AND recom_rating>=%s"
                        cursor.execute(dbQuery, (user_id, recom_rating))
                    else:
                        dbQuery = "SELECT * FROM recommendation WHERE user_id=%s"
                        cursor.execute(dbQuery, (user_id))
                self.db_connector.commit()
                param_list = ["recom_id", "user_id", "rm_id", "recom_rating"]
                for row in cursor:
                    rm = self._parse_result(param_list, row, underscore=False)
                    recomList.append(rm)
                # user["birthday"] = user["birthday"].strftime('%Y-%m-%d')
            except Exception as e:
                print("Exception: ", e)

        return recomList

    def get_feedbacks(self, as_id):
        fb_list = []
        with self.db_connector.cursor() as cursor:
            try:
                dbQuery = "SELECT * FROM feedback WHERE as_id=%s"
                cursor.execute(dbQuery, as_id)
                self.db_connector.commit()
                param_list = ["id", "feedback_type", "as_id", "m_id", "rating", "rated_date"]
                for row in cursor:
                    feedback = self._parse_result(param_list, row, underscore=False)
                    fb_list.append(feedback)
                # user["birthday"] = user["birthday"].strftime('%Y-%m-%d')
            except Exception as e:
                print("Exception: ", e)

        return fb_list
