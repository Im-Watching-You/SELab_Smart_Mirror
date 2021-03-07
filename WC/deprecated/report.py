import WC.dbconnector as db
from datetime import datetime


class Report:

    def __init__(self):
        self.conn = None

    def retrieve_user_report(self, user_id, dict_duration=None):

        result = self.__db_retrieve_report('user_spaerf', 'user_id', user_id, 'taken_date', dict_duration)

        if result:
            return result
        else:
            return False

    def retrieve_aging_model_report(self, model_id, dict_duration=None):

        result = self.__db_retrieve_report('aging_maf', 'model_id', model_id, 'rated_date', dict_duration)

        if result:
            return result
        else:
            return False

    def retrieve_emotion_model_report(self, model_id, dict_duration=None):

        result = self.__db_retrieve_report('emotion_maf', 'model_id', model_id, 'rated_date', dict_duration)

        if result:
            return result
        else:
            return False

    def retrieve_recommendation_model_report(self, model_id, dict_duration=None):

        result = self.__db_retrieve_report('recommendation_maf', 'model_id', model_id, 'rated_date', dict_duration)

        if result:
            return result
        else:
            return False

    @staticmethod
    def dict_duration_to_sql(dict_duration):

        start_date = '2000-01-01'
        end_date = '2099-12-31'

        if 'start_date' in dict_duration:
            start_date = datetime.strptime(dict_duration['start_date'], "%Y%m%d")

        if 'end_date' in dict_duration:
            end_date = datetime.strptime(dict_duration['end_date'], "%Y%m%d")


        return f"BETWEEN \'{start_date}\' AND \'{end_date}\'"

    def __db_retrieve_report(self, table_type, id_type, id_value, date, dict_duration=None):

        if dict_duration:

            sql = f"SELECT * FROM smart_mirror_system.{table_type} WHERE {id_type} = \'{id_value}\' " \
                f"AND {date} " + self.dict_duration_to_sql(dict_duration)

        else:
            sql = f"SELECT * FROM smart_mirror_system.{table_type} WHERE {id_type} = \'{id_value}\'"

        db.connect_to_db(self)
        with self.conn.cursor() as cursor:
            try:
                cursor.execute(sql)
                result = cursor.fetchall()
                db.disconnect_from_db(self)
                return result

            except Exception as e:
                print(e)
                db.disconnect_from_db(self)
                return False


if __name__ == '__main__':
    rpt = Report()
    # rpt.retrieve_user_report(60, {"start_date": '20190625'})
    rpt.retrieve_aging_model_report(1)
    # rpt.retrieve_aging_model_report(model_id=1, dict_duration={"start_date": '20190628', "end_date": '20190702'})
