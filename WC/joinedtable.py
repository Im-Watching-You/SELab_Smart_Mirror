import WC.dbconnector as db
from datetime import timedelta


class JoinedTable:
    pass


class SPATable(JoinedTable):
    """
    Using user_spa :View (Session, Photo, Assessment)
    """
    def __init__(self):
        self.conn = None
        db.connect_to_db(self)

    def __del__(self):
        db.disconnect_from_db(self)

    #origin code
    def retrieve_user_variation(self, user_id, assess_type, table_type, dict_duration):

        max_gap = (dict_duration['end_date'] - dict_duration['start_date']).days

        # print(max_gap)

        return self._db_retrieve_user_variation(user_id, assess_type, table_type, dict_duration, max_gap)

    #tmp code
    def retrieve_user_variation_tmp(self, user_id, assess_type, dict_duration):

        max_gap = (dict_duration['end_date'] - dict_duration['start_date']).days

        return self._db_retrieve_user_variation_tmp(user_id, assess_type, dict_duration, max_gap)

    def retrieve_user_trend(self, user_id, assess_type, dict_duration):
        return self._db_retrieve_user_trend(user_id, assess_type, dict_duration)

    def retrieve_user_view(self, user_id, assess_type, table_type):
        """

        :param user_id:
        :param assess_type:
        :param table_type:
        :return:
        """
        if assess_type not in ['aging_diagnosis', 'emotion_diagnosis']:
            print("ERROR: WC.joined table.retrieve_user_view : invalid assess type")
            return []
        elif table_type not in ['spa', 'spaaf', 'spaef']:
            print("ERROR: WC.joined table.retrieve_user_view : invalid table type")
            return []
        return self._db_retrieve_user_view(user_id=user_id, assess_type=assess_type, table_type=table_type)

    # origin code
    def _db_retrieve_user_variation(self, user_id, assess_type, table_type, dict_duration, max_gap):

        sql_start = f"SELECT * FROM smart_mirror_system.user_{table_type} " \
                    f"WHERE user_id = \'{user_id}\' " \
                    f"AND assess_type = \'{assess_type}\' " \
                    f"AND recorded_date >= \'{dict_duration['start_date']}\' " \
                    f"AND recorded_date < \'{dict_duration['start_date'] + timedelta(days=1)}\'"

        sql_end = f"SELECT * FROM smart_mirror_system.user_{table_type} " \
                  f"WHERE user_id = \'{user_id}\' " \
                  f"AND assess_type = \'{assess_type}\' " \
                  f"AND recorded_date >= \'{dict_duration['end_date']}\' " \
                  f"AND recorded_date < \'{dict_duration['end_date'] + timedelta(days=1)}\'"

        # print(sql_start)
        # print(sql_end)

        # max gap between two points, when gap is bigger than 'max_gap' stop retrieve and return [],[]

        with self.conn.cursor() as cursor:
            try:
                cursor.execute(sql_start)
                start_date_rows = cursor.fetchall()
                # print(start_date_rows)

                cursor.execute(sql_end)

                end_date_rows = cursor.fetchall()

                if start_date_rows != () and end_date_rows != ():
                    # print(dict_duration)
                    return start_date_rows, end_date_rows

                elif start_date_rows == () and end_date_rows == ():
                    if dict_duration['start_date'] == dict_duration['end_date'] - timedelta(days=1) or dict_duration['start_date'] == dict_duration['end_date']:
                        # print(f"We've found all data of date within given duration, "
                        #       f"but there were not enough data for analyzing variation.")
                        return [], []
                    # print(dict_duration)
                    dict_duration['start_date'] += timedelta(days=1)
                    dict_duration['end_date'] -= timedelta(days=1)
                    return self._db_retrieve_user_variation(user_id, assess_type, table_type, dict_duration, max_gap)

                elif start_date_rows == () and end_date_rows != ():
                    if dict_duration['start_date'] == dict_duration['end_date'] - timedelta(days=1):
                        # print(f"We've found all data of date within given duration, "
                        #       f"but there were not enough data for analyzing variation.")
                        return [], []
                    # print(dict_duration)
                    dict_duration['start_date'] += timedelta(days=1)
                    return self._db_retrieve_user_variation(user_id, assess_type, table_type, dict_duration, max_gap)

                elif start_date_rows != () and end_date_rows == ():
                    if dict_duration['start_date'] == dict_duration['end_date'] - timedelta(days=1):
                        # print(f"We've found all data of date within given duration, "
                        #       f"but there were not enough data for analyzing variation.")
                        return [], []
                    # print(dict_duration)
                    dict_duration['end_date'] -= timedelta(days=1)
                    return self._db_retrieve_user_variation(user_id, assess_type, table_type, dict_duration, max_gap)

            except Exception as e:
                print(e)
                print("EXCEPTION: WC.joined table._db_retrieve_user_variation")
                return [], []

    # tmp code
    def _db_retrieve_user_variation_tmp(self, user_id, assess_type, dict_duration, max_gap):

        sql_start = f"SELECT * FROM smart_mirror_system.user_spa " \
                    f"WHERE user_id = \'{user_id}\' " \
                    f"AND assess_type = \'{assess_type}\' " \
                    f"AND recorded_date >= \'{dict_duration['start_date']}\' " \
                    f"AND recorded_date < \'{dict_duration['start_date'] + timedelta(days=1)}\'"

        sql_end = f"SELECT * FROM smart_mirror_system.user_spa " \
                  f"WHERE user_id = \'{user_id}\' " \
                  f"AND assess_type = \'{assess_type}\' " \
                  f"AND recorded_date >= \'{dict_duration['end_date']}\' " \
                  f"AND recorded_date < \'{dict_duration['end_date'] + timedelta(days=1)}\'"

        # print(sql_start)
        # print(sql_end)
        # max gap between two points, when gap is bigger than 'max_gap' stop retrieve and return [],[]

        # print(max_gap)

        with self.conn.cursor() as cursor:
            try:
                cursor.execute(sql_start)
                start_date_rows = cursor.fetchall()
                # print(start_date_rows)

                cursor.execute(sql_end)

                end_date_rows = cursor.fetchall()
                # print(end_date_rows)

                if start_date_rows != () and end_date_rows != ():
                    # print(dict_duration)
                    return start_date_rows, end_date_rows

                elif start_date_rows == () and end_date_rows == ():
                    if dict_duration['start_date'] == dict_duration['end_date'] - timedelta(days=1) or dict_duration['start_date'] == dict_duration['end_date']:
                        # print(f"We've found all data of date within given duration, "
                        #       f"but there were not enough data for analyzing variation.")
                        return [], []
                    # print(dict_duration)
                    dict_duration['start_date'] += timedelta(days=1)
                    dict_duration['end_date'] -= timedelta(days=1)
                    return self._db_retrieve_user_variation_tmp(user_id, assess_type, dict_duration, max_gap)

                elif start_date_rows == () and end_date_rows != ():
                    if dict_duration['start_date'] == dict_duration['end_date'] - timedelta(days=1):
                        # print(f"We've found all data of date within given duration, "
                        #       f"but there were not enough data for analyzing variation.")
                        return [], []
                    # print(dict_duration)
                    dict_duration['start_date'] += timedelta(days=1)
                    return self._db_retrieve_user_variation_tmp(user_id, assess_type, dict_duration, max_gap)

                elif start_date_rows != () and end_date_rows == ():
                    if dict_duration['start_date'] == dict_duration['end_date'] - timedelta(days=1):
                        # print(f"We've found all data of date within given duration, "
                        #       f"but there were not enough data for analyzing variation.")
                        return [], []
                    # print(dict_duration)
                    dict_duration['end_date'] -= timedelta(days=1)
                    return self._db_retrieve_user_variation_tmp(user_id, assess_type, dict_duration, max_gap)

            except Exception as e:
                print(e)
                print("EXCEPTION: WC.joined table._db_retrieve_user_variation")
                return [], []

    def _db_retrieve_user_trend(self, user_id, assess_type, dict_duration):

        if assess_type == 'aging_diagnosis':

            sql = f"SELECT result, Age_Wrinkle, Age_Spot, Age_Geo, recorded_date FROM smart_mirror_system.user_spaaf " \
              f"WHERE user_id = \'{user_id}\' " \
              f"AND assess_type = \'{assess_type}\' " \
              f"AND recorded_date >= \'{dict_duration['start_date']}\' " \
              f"AND recorded_date < \'{dict_duration['end_date'] + timedelta(days=1)}\'"

        elif assess_type == 'emotion_diagnosis':

            sql = f"SELECT * FROM smart_mirror_system.user_spaef " \
              f"WHERE user_id = \'{user_id}\' " \
              f"AND assess_type = \'{assess_type}\' " \
              f"AND recorded_date >= \'{dict_duration['start_date']}\' " \
              f"AND recorded_date < \'{dict_duration['end_date'] + timedelta(days=1)}\'"

        with self.conn.cursor() as cursor:
            try:
                # print(dict_duration)
                # print(sql)
                cursor.execute(sql)
                result = cursor.fetchall()
                # print(len(result))
                # print(result)

                if result is ():
                    return []
                else:
                    return result

            except Exception as e:
                print(e)
                print("ERROR: WC.joinedtable._db_retrieve_user_trend")
                return []

    # for testing time series, to get raw table from db
    def _db_retrieve_user_view(self, user_id, assess_type, table_type):
        """
        Testing Method
        :param user_id: Integer
        :param assess_type: ['aging_diagnosis' or 'emotion_diagnosis']
        :param table_type: ['spa' or 'spaf']
        :return: list of dictionaries
        """

        sql = f"SELECT * FROM smart_mirror_system.user_{table_type} " \
              f"WHERE user_id = \'{user_id}\' " \
              f"AND assess_type = \'{assess_type}\' "

        with self.conn.cursor() as cursor:
            try:
                # print(sql)
                cursor.execute(sql)
                result = cursor.fetchall()

                if result is ():
                    print("No result found.")
                    return []
                else:
                    return result

            except Exception as e:
                print(e)
                print("ERROR: WC.joinedtable._db_retrieve_user_trend")
                return []


if __name__ == '__main__':
    spa = SPATable()
    # result = spa.retrieve_user_assessment(user_id=294, assess_type='aging_diagnosis', duration='monthly')
    # print(len(result), result, sep='\n')

    # spa._db_retrieve_user_assessment(10,'aging_diagnosis',{"start_date":datetime.now().date(),'end_date':datetime.now().date()},'variation')
