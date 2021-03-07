import WC.dbconnector as db
from datetime import timedelta
from datetime import datetime
from dateutil.relativedelta import relativedelta


class SettingManager:
    pass


class SystemMonitor:
    pass


class DatabaseMonitor(SystemMonitor):

    def retrieve_user_table_info(self):
        """

        :return:Dictionary
        """
        # SQL list
        sql_male_count = "SELECT count(*), avg(age) as count FROM smart_mirror_system.user WHERE gender = 'male'"
        sql_female_count = "SELECT count(*), avg(age) as count FROM smart_mirror_system.user WHERE gender = 'female'"

        db.connect_to_db(self)
        with self.conn.cursor() as cursor:
            try:
                cursor.execute(sql_male_count)
                result = cursor.fetchone()
                num_of_male, avg_age_of_male = result.values()

                cursor.execute(sql_female_count)
                result = cursor.fetchone()
                num_of_female, avg_age_of_female = result.values()

            except Exception as e:
                print(e)
                print("ERROR : __db_user_brief_info")
                db.disconnect_from_db(self)
                return []

        return {"total_num_of_user": num_of_male + num_of_male,
                "avg_age_of_user": round(float((avg_age_of_male + avg_age_of_female) / 2), 2),
                "num_of_male": num_of_male, "avg_age_of_male": round(float(avg_age_of_male), 2),
                "num_of_female": num_of_female, "avg_age_of_female": round(float(avg_age_of_female), 2)}

    def retrieve_staff_table_info(self):
        """

        :return:Dictionary
        """
        # SQL list
        sql_count_male = "SELECT count(*), avg(age) as count FROM smart_mirror_system.staff WHERE gender = 'male'"
        sql_count_female = "SELECT count(*), avg(age) as count FROM smart_mirror_system.staff WHERE gender = 'female'"

        db.connect_to_db(self)
        with self.conn.cursor() as cursor:
            try:
                cursor.execute(sql_count_male)
                result = cursor.fetchone()
                num_of_male, avg_age_of_male = result.values()

                cursor.execute(sql_count_female)
                result = cursor.fetchone()
                num_of_female, avg_age_of_female = result.values()

            except Exception as e:
                print(e)
                print("ERROR : __db_staff_brief_info")
                db.disconnect_from_db(self)
                return []

        return {"total_num_of_user": num_of_male + num_of_male,
                "avg_age_of_user": round(float((avg_age_of_male + avg_age_of_female) / 2), 2),
                "num_of_male": num_of_male, "avg_age_of_male": round(float(avg_age_of_male), 2),
                "num_of_female": num_of_female, "avg_age_of_female": round(float(avg_age_of_female), 2)}

    def retrieve_session_table_info(self):

        # total number of session
        sql_count_session = "SELECT count(*) as count FROM smart_mirror_system.session"

        # current number of log in User
        sql_count_log_in_user = "SELECT count(*) as count FROM smart_mirror_system.session " \
                                "where u_id is not null and log_out_date is null"

        # The most visited user
        sql_find_most_visited_user = "SELECT u_id, count(*) as count FROM smart_mirror_system.session " \
                                   "GROUP BY u_id ORDER BY count DESC limit 1"

        db.connect_to_db(self)
        with self.conn.cursor() as cursor:
            try:
                cursor.execute(sql_count_session)
                result = cursor.fetchone()
                total_num_of_session = result['count']
                # print(total_num_of_session)

                cursor.execute(sql_count_log_in_user)
                result = cursor.fetchone()
                num_of_log_in_user = result['count']
                # print(num_of_log_in_user)

                cursor.execute(sql_find_most_visited_user)
                result = cursor.fetchone()
                user_id, count = result.values()
                # print(user_id, count)

            except Exception as e:
                print(e)
                print("ERROR : __db_session_brief_info")
                db.disconnect_from_db(self)
                return []

        return {"total_num_of_session": total_num_of_session,
                "num_of_current log in User": num_of_log_in_user,
                "most_visited_user": user_id, "num_of_highest_visits": count}

    def retrieve_photo_table_info(self):
        # total number of photos
        sql_count_photo = "SELECT count(*) as count FROM smart_mirror_system.photo"

        # number of photos taken in a week
        sql_count_photo_in_a_week = f"SELECT count(*) as count FROM smart_mirror_system.photo " \
                                    f"WHERE taken_date BETWEEN \'{datetime.now()-timedelta(days=7)}\' " \
                                    f"AND \'{datetime.now()}\'"

        # number of photos taken in a month
        sql_count_photo_in_a_month = f"SELECT count(*) as count FROM smart_mirror_system.photo " \
                                     f"WHERE taken_date BETWEEN \'{datetime.now()-relativedelta(months=1)}\' " \
                                     f"AND \'{datetime.now()}\'"

        db.connect_to_db(self)
        with self.conn.cursor() as cursor:
            try:
                cursor.execute(sql_count_photo)
                result = cursor.fetchone()
                total_num_of_photo = result['count']
                # print(sql_count_photo)

                cursor.execute(sql_count_photo_in_a_week)
                result = cursor.fetchone()
                num_of_photos_in_a_week = result['count']
                # print(sql_count_photo_in_a_week)

                cursor.execute(sql_count_photo_in_a_month)
                result = cursor.fetchone()
                num_of_photos_in_a_month = result['count']
                # print(sql_count_photo_in_a_month)

            except Exception as e:
                print(e)
                print("ERROR : retrieve_photo_table_info")
                db.disconnect_from_db(self)
                return []

        return {"total_num_of_photos": total_num_of_photo,
                "num_of_photos_in_a_week": num_of_photos_in_a_week,
                "num_of_photos_in_a_month": num_of_photos_in_a_month}

    def retrieve_assessment_table_info(self):

        # TODO: Diagnosis 통합시키면 코드 수정
        # total number of assessments
        sql_count_assessment = "SELECT count(*) as count FROM smart_mirror_system.assessment"

        # number of assessments in a week
        sql_count_assessment_in_a_week = f"SELECT count(*) as count FROM smart_mirror_system.photo " \
                                    f"WHERE taken_date BETWEEN \'{datetime.now()-timedelta(days=7)}\' " \
                                    f"AND \'{datetime.now()}\'"

        # number of assessments in a month
        sql_count_assessment_in_a_month = f"SELECT count(*) as count FROM smart_mirror_system.photo " \
                                     f"WHERE taken_date BETWEEN \'{datetime.now()-relativedelta(months=1)}\' " \
                                     f"AND \'{datetime.now()}\'"

        db.connect_to_db(self)
        with self.conn.cursor() as cursor:
            try:
                cursor.execute(sql_count_photo)
                result = cursor.fetchone()
                total_num_of_photo = result['count']
                # print(sql_count_photo)

                cursor.execute(sql_count_photo_in_a_week)
                result = cursor.fetchone()
                num_of_photos_in_a_week = result['count']
                # print(sql_count_photo_in_a_week)

                cursor.execute(sql_count_photo_in_a_month)
                result = cursor.fetchone()
                num_of_photos_in_a_month = result['count']
                # print(sql_count_photo_in_a_month)

            except Exception as e:
                print(e)
                print("ERROR : retrieve_photo_table_info")
                db.disconnect_from_db(self)
                return []

        return {"total_num_of_photos": total_num_of_photo,
                "num_of_photos_in_a_week": num_of_photos_in_a_week,
                "num_of_photos_in_a_month": num_of_photos_in_a_month}

    def retrieve_feedback_table_info(self):
       pass

    def retrieve_training_set_table_info(self):
       pass

    def retrieve_AAAApp_table_info(self):
        pass


class ModelMonitor(SystemMonitor):

    def retrieve_aging_model_info(self):
        model_type = 'aging_model'
        sql_count_model = f"SELECT Count(*) AS count FROM smart_mirror_system.ml_model " \
                          f"WHERE model_type = \'{model_type}\'"

        # It contains average of number of data, average of accuracy
        sql_avg_data_of_aging_models = f"SELECT model_type, avg(accuracy), avg(num_of_data) " \
                                        f"FROM smart_mirror_system.ml_model WHERE model_type = \'{model_type}\'"

    def retrieve_emotion_model_info(self):
        model_type = 'emotion_model'
        sql_count_model = f"SELECT Count(*) AS count FROM smart_mirror_system.ml_model " \
                          f"WHERE model_type = \'{model_type}\'"

        # It contains average of number of data, average of accuracy
        sql_avg_data_of_emotion_model = f"SELECT model_type, avg(accuracy), avg(num_of_data) " \
                                        f"FROM smart_mirror_system.ml_model WHERE model_type = \'{model_type}\'"

    def retrieve_recommendation_model_info(self):
        model_type = 'recommendation_model'
        sql_count_model = f"SELECT Count(*) AS count FROM smart_mirror_system.ml_model " \
                          f"WHERE model_type = \'{model_type}\'"

        # It contains average of number of data, average of accuracy
        sql_avg_data_of_emotion_model = f"SELECT model_type, avg(accuracy), avg(num_of_data) " \
                                        f"FROM smart_mirror_system.ml_model WHERE model_type = \'{model_type}\'"

    def retrieve_face_detection_model_info(self):
        model_type = 'face_detection_model'
        sql_count_model = f"SELECT Count(*) AS count FROM smart_mirror_system.ml_model " \
                          f"WHERE model_type = \'{model_type}\'"

        # It contains average of number of data, average of accuracy
        sql_avg_data_of_emotion_model = f"SELECT model_type, avg(accuracy), avg(num_of_data) " \
                                        f"FROM smart_mirror_system.ml_model WHERE model_type = \'{model_type}\'"


class BackUpManager:

    def export_tables(self, table_type=None, save_path=None):
        pass

    def export_models(self, model_type=None, save_path=None):
        pass

    def export_photos(self, save_path = None):
        pass


if __name__ == '__main__':
    dbm = DatabaseMonitor()

    # dbd.get_user_brief_info()
    # dbd.get_staff_brief_info()

    # print(dbm.retrieve_user_table_info())
    print(dbm.retrieve_photo_table_info())