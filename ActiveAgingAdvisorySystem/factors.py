import ActiveAgingAdvisorySystem.dbconnector as db
import ActiveAgingAdvisorySystem.stringmaker as sm

from datetime import datetime


class Factor:

    def __init__(self):
        self.conn = None
        self.num_of_emotion_factors = 11

        db.connect_to_db(self)

    def __del__(self):
        db.disconnect_from_db(self)

    def register_emotion_factor(self, assess_id, dict_info):
        """
        :param assess_id: id of assessment table row
        :param dict_info: dictionary of factor list {'(factor_name_1)': (value), '(factor_name_2)': (value)...}
        :return: boolean

        TestCode
            ft = Factor()
            ft.register_emotion_factor(assess_id=12347, dict_info={'f333': 1, 'f3': 2})
        """

        return self._db_register_emotion_factor(assess_id, dict_info)

    def register_age_factor(self, assess_id, dict_info):
        """
        :param assess_id: id of assessment table row
        :param dict_info: dictionary of factor list {'(factor_name_1)': (value), '(factor_name_2)': (value)...}
        :return: boolean

        TestCode
            ft = Factor()
            ft.register_age_factor(assess_id=12347, dict_info={'f333': 1, 'f3': 2})
        """

        return self._db_register_age_factor(assess_id, dict_info)

    def _db_register_emotion_factor(self, assess_id, dict_info):

        avail_factor_name = ['f' + str(i) for i in range(1, self.num_of_emotion_factors + 1)]
        avail_factor_name.append('emotion')

        factor_name = list(dict_info.keys())

        for i in factor_name:
            if i not in avail_factor_name:
                print(f"Invalid factor name: \'{i}\'")
                return False

        factor_value = list(str(i) for i in dict_info.values())

        sql = f"INSERT INTO emotion_factor (assess_id, recorded_date, " + sm.StringMaker().list_to_string_with_comma(factor_name) + ") " \
              f"VALUES ({assess_id}, \'{datetime.now()}\', "+sm.StringMaker().list_to_string_with_comma_quote(factor_value)+")"

        print(sql)
        db.connect_to_db(self)
        with self.conn.cursor() as cursor:
            try:
                cursor.execute(sql)
                db.disconnect_from_db(self)
                return True

            except Exception as e:
                print(e)
                print("ERROR : WC.factor._db_register_emotion_factor")
                db.disconnect_from_db(self)
                return False

    def _db_register_age_factor(self, assess_id, dict_info):

        avail_factor_name = ["assess_id",
                             "Density_Corner_Left_Eye",
                             "Density_Corner_Right_Eye",
                             "Density_Forehead",
                             "Density_Cheek_Left",
                             "Density_Cheek_Right",

                             "Depth_Corner_Left_Eye",
                             "Depth_Corner_Right_Eye",
                             "Depth_Forehead",
                             "Depth_Cheek_Left",
                             "Depth_Cheek_Right",

                             "Variance_Corner_Left_Eye",
                             "Variance_Corner_Right_Eye",
                             "Variance_Forehead",
                             "Variance_Cheek_Left",
                             "Variance_Cheek_Right",
                             "Age_Wrinkle",

                             "Spot_Density_Corner_Left_Eye",
                             "Spot_Density_Corner_Right_Eye",
                             "Spot_Density_Forehead",
                             "Spot_Density_Cheek_Left",
                             "Spot_Density_Cheek_Right",

                             "Max_Spot_Size_Corner_Left_Eye",
                             "Max_Spot_Size_Corner_Right_Eye",
                             "Max_Spot_Size_Forehead",
                             "Max_Spot_Size_Cheek_Left",
                             "Max_Spot_Size_Cheek_Right",

                             "Count_of_Spots_Corner_Left_Eye",
                             "Count_of_Spots_Corner_Right_Eye",
                             "Count_of_Spots_Forehead",
                             "Count_of_Spots_Cheek_Left",
                             "Count_of_Spots_Cheek_Right",
                             "Age_Spot",

                             "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11", "F12",
                             "Age_Geo", "Recorded_date"]

        factor_name = list(dict_info.keys())

        for i in factor_name:
            if i not in avail_factor_name:
                print(f"Invalid factor name: \'{i}\'")
                return False

        factor_value = list(str(i) for i in dict_info.values())

        sql = f"INSERT INTO smart_mirror_system.age_factor (assess_id, recorded_date, " + sm.StringMaker().list_to_string_with_comma(
            factor_name) + ") " \
                  f"VALUES ({assess_id}, \'{datetime.now()}\', " + sm.StringMaker().list_to_string_with_comma_quote(factor_value) + ")"

        # print(sql)
        db.connect_to_db(self)
        with self.conn.cursor() as cursor:
            try:
                cursor.execute(sql)
                db.disconnect_from_db(self)
                return True

            except Exception as e:
                print(e)
                print("ERROR : WC.factor._db_register_age_factor")
                db.disconnect_from_db(self)
                return False

if __name__ == '__main__':
    ft = Factor()
    # ft.register_emotion_factor(assess_id=123557, dict_info={'emotion': 3, 'f3': float(0.2)})
    ft.register_age_factor(assess_id=151, dict_info={'Density_Corner_Left_Eye': 0.1})
