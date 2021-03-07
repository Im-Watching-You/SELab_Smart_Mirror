import WC.dbconnector as db
import WC.stringmaker as sm
import random
from datetime import datetime


class Feedback:
    def __init__(self):
        self.coon = None
        db.connect_to_db(self)

    def __del__(self):
        db.disconnect_from_db(self)

    def _db_register_feedback(self, feedback_type, dict_info):
        """
        :param feedback_type:
            String, a type of assessment in ("aging_feedback", "emotion_feedback", "recommendation_feedback")
        :param dict_info: Dictionary
            key(1): assessment_id: Integer
            key(2): model_id: Integer
            key(3): rating: Integer
        :return:boolean
        """

        for i in list(dict_info):
            if i is 'assessment_id':
                as_id = dict_info[i]
            elif i is 'model_id':
                m_id = dict_info[i]
            elif i is 'rating':
                rating = dict_info[i]

        try:
            sql = f"INSERT INTO smart_mirror_system.feedback (feedback_type, as_id, m_id, rating, rated_date)" \
                f"VALUES (\'{feedback_type}\', {as_id}, {m_id}, {rating}, \'{datetime.now()}\')"
        except UnboundLocalError:
            print("You should check your key spelling. Key should be one of ['assessment_id','model_id', 'rating']")
            return False


        with self.conn.cursor() as cursor:
            try:
                cursor.execute(sql)
                # print(sql)
                print("Feedback data has been inserted.")

                return True

            except Exception as e:
                print(e)

                return False

    def _db_retrieve_feedback_by_ids(self, feedback_type, dict_condition=None):
        """

        :param feedback_type: String, one of ['aging_feedback', 'emotion_feedback', 'recommendation_feedback']
        :param dict_condition:
            key(1) id: Integer
            key(2) assessment_id: Integer
            key(3) model_id: Integer
        :return:
        """

        # To make dict_condition > {'id': None, 'as_id': None, 'm_id': None}
        if dict_condition is None:
            dict_condition = dict.fromkeys(['id', 'assessment_id', 'model_id'])

        id, as_id, m_id, condition = None, None, None, ''

        for i in list(dict_condition):
            if i is 'id':
                id = dict_condition[i]
            elif i is 'assessment_id':
                as_id = dict_condition[i]
            elif i is 'model_id':
                m_id = dict_condition[i]
            else:
                print("Please check your key name again.")
                print(f"\'{i}\' is invalid.")
                return []

        sql = f"SELECT * FROM smart_mirror_system.feedback WHERE feedback_type = \'{feedback_type}\' "

        if id is None and as_id is None and m_id is None:
            condition = ""
        elif id is not None and as_id is None and m_id is None:
            condition = f"AND id = \'{id}\'"
        elif id is not None and as_id is not None and m_id is None:
            condition = f"AND id = \'{id}\' AND as_id = \'{as_id}\'"
        elif id is not None and as_id is None and m_id is not None:
            condition = f"AND id = \'{id}\' AND m_id = \'{m_id}\'"

        elif id is not None and as_id is not None and m_id is not None:
            condition = f"AND id = \'{id}\' AND as_id = \'{as_id}\' AND m_id = \'{m_id}\'"
        elif as_id is not None and m_id is not None:
            condition = f"AND as_id = \'{as_id}\' and m_id = \'{m_id}\'"
        elif as_id is not None and m_id is None:
            condition = f"AND as_id = \'{as_id}\'"
        elif as_id is None and m_id is not None:
            condition = f"AND m_id = \'{m_id}\'"

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
                print("_db_retrieve_feedback_by_ids : error")

                return []

    def _db_retrieve_feedback_by_rating(self, feedback_type, min_rating=None, max_rating=None, sorting_order=None):

        sql = f"SELECT * FROM smart_mirror_system.feedback " \
              f"WHERE rating >= {min_rating} AND rating <= {max_rating} AND feedback_type = \'{feedback_type}\' " \
              f"ORDER BY rating {sorting_order}"

        # print(sql)

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
                print("_db_retrieve_feedback_by_rating : error" )
                return []

    def _db_retrieve_feedback_by_recorded_date(self, feedback_type, start_date='20000101', end_date='20991231'):

        sql = f"SELECT * FROM smart_mirror_system.feedback " \
              f"WHERE recorded_date >= \'{start_date}\' AND recorded_date <= \'{end_date}\' " \
              f"AND feedback_type = \'{feedback_type}\'"

        db.connect_to_db(self)
        with self.conn.cursor() as cursor:
            try:
                cursor.execute(sql)
                result = cursor.fetchall()

                if result is ():
                    return []

            except Exception as e:
                print(e)
                print("_db_retrieve_feedback_by_recorded_date : error")
                return []

    def _db_update_feedback(self, feedback_type, id, dict_info):
        """
        :param feedback_type:
            String, a type of assessment in ("aging_feedback", "emotion_feedback", "recommendation_feedback")
        :param id: Integer
        :param dict_info: Any attributes that you want to update
        :return: Boolean
        """
        db.connect_to_db(self)
        with self.conn.cursor() as cursor:

            # To check dict_info has format problem
            condition_string = sm.StringMaker().key_equation_value(dict_info)
            if condition_string is "":
                print("Invalid input in dict_info.")
                return False

            # To validate dict_info's keys
            for i in dict_info:
                if i not in ('as_id', 'm_id', 'rating', 'rated_date'):
                    print(f"Invalid key type \'{i}\'. Please check your key again.")
                    return False

            # To update feedback

            sql = f"UPDATE smart_mirror_system.feedback SET {condition_string} " \
                  f"WHERE id = {id} AND feedback_type = \'{feedback_type}\'"

            try:
                # cursor.execute() returns number of afftected rows
                is_updated = cursor.execute(sql)

                if is_updated:
                    print("Feedback data has benn updated.")
                else:
                    print(f"There is no matching id or Input data has already been applied.")
                return True

            except Exception as e:
                print(e)
                print("ERROR: WC.feedback._db_update_feedback")
                return False

    def _db_make_dummy_feedback(self, feedback_type, count):

        db.connect_to_db(self)
        with self.conn.cursor() as cursor:
            for _ in range(count):
                try:
                    sql = f"INSERT INTO feedback (feedback_type, as_id, m_id, rating, rated_date)" \
                          f"VALUES (\'{feedback_type}\', {random.randrange(1,45)}, " \
                          f"{random.randrange(101,140)}, {random.randrange(1,5)}, \'{datetime.now()}\')"
                    cursor.execute(sql)

                    print("Feedback data has been inserted.")

                except Exception as e:
                    print(e)
                    print('ERROR: _db_make_dummy')
                    return False

            return True

    def _db_truncate_feedback(self, feedback_type):

        option = input(f'<STAFF ONLY> This method will delete entire rows of \'{feedback_type}\' data. Are you Sure? (Y/N)')

        if option.lower() == 'y':
            db.connect_to_db(self)
            with self.conn.cursor() as cursor:
                try:
                    sql = f"DELETE FROM smart_mirror_system.feedback WHERE feedback_type = \'{feedback_type}\'"
                    cursor.execute(sql)
                    print(f"All \'{feedback_type}\' type data has been truncated.")
                    return True

                except Exception as e:
                    print(e)
                    return False
        else:
            print('Truncate Canceled.')
            return False


class AgingFeedback(Feedback):

    feedback_type = 'aging_feedback'

    def register_feedback(self, dict_info):
        """

        :param dict_info: Dictionary
            key(1): assessment_id: Integer
            key(2): model_id: Integer
            key(3): rating: Integer
        :return:boolean
        """
        return self._db_register_feedback(self.feedback_type, dict_info)

    def retrieve_feedback_by_ids(self, dict_condition=None):
        """

        :param dict_condition:
            key(1) id: Integer
            key(2) assessment_id: Integer
            key(3) model_id: Integer
        :return:
        """
        return self._db_retrieve_feedback_by_ids(self.feedback_type, dict_condition)

    def retrieve_feedback_by_rating(self, dict_condition):
        """

        :param dict_condition: {"min_rating"  : Integer, "max_rating" : Integer, "sorting_order" : ASC or DESC}
        :return: List of Dictionaries
        """

        # Default Value
        min_rating = 0
        max_rating = 5
        sorting_order = "DESC"

        for i in list(dict_condition):
            if i is 'min_rating':
                min_rating = dict_condition[i]
            elif i is 'max_rating':
                max_rating = dict_condition[i]
            elif i is 'sorting_order':
                sorting_order = dict_condition[i]

        return self._db_retrieve_feedback_by_rating(self.feedback_type, min_rating= min_rating,
                                                    max_rating=max_rating, sorting_order=sorting_order)

    def update_feedback(self, id, dict_info):
        """

        :param id: Integer: feedback_id
        :param dict_info: any attributes want to change (feedback_type, as_id, m_id, rating, rated_date)
        :return:
        """
        return self._db_update_feedback(self.feedback_type, id, dict_info)

    def make_dummy_feedback(self, count=20):
        return self._db_make_dummy_feedback(self.feedback_type, count)

    def truncate_feedback(self):
        return self._db_truncate_feedback(feedback_type=self.feedback_type)


class EmotionFeedback(Feedback):

    feedback_type = 'emotion_feedback'

    def register_feedback(self, dict_info):
        """

        :param dict_info: Dictionary
            key(1): assessment_id: Integer
            key(2): model_id: Integer
            key(3): rating: Integer
        :return:boolean
        """
        return self._db_register_feedback(self.feedback_type, dict_info)

    def retrieve_feedback_by_ids(self, dict_condition=None):
        """

        :param dict_condition:
            key(1) id: Integer
            key(2) assessment_id: Integer
            key(3) model_id: Integer
        :return:
        """
        return self._db_retrieve_feedback_by_ids(self.feedback_type, dict_condition)

    def retrieve_feedback_by_rating(self, dict_condition):
        """

        :param dict_condition: {"min_rating"  : Integer, "max_rating" : Integer, "sorting_order" : ASC or DESC}
        :return: List of Dictionaries
        """

        # Default Value
        min_rating = 0
        max_rating = 5
        sorting_order = "DESC"

        for i in list(dict_condition):
            if i is 'min_rating':
                min_rating = dict_condition[i]
            elif i is 'max_rating':
                max_rating = dict_condition[i]
            elif i is 'sorting_order':
                sorting_order = dict_condition[i]

        return self._db_retrieve_feedback_by_rating(self.feedback_type, min_rating=min_rating,
                                                      max_rating=max_rating, sorting_order=sorting_order)

    def update_feedback(self, id, dict_info):
        """

        :param id: Integer: feedback_id
        :param dict_info: any attributes want to change (feedback_type, as_id, m_id, rating, rated_date)
        :return:
        """
        return self._db_update_feedback(self.feedback_type, id, dict_info)

    def make_dummy_feedback(self, count=20):
        return self._db_make_dummy_feedback(self.feedback_type, count)

    def truncate_feedback(self):
        return self._db_truncate_feedback(feedback_type=self.feedback_type)


class RecommendationFeedback(Feedback):

    feedback_type = 'recommendation_feedback'

    def register_feedback(self, dict_info):
        """

        :param dict_info: Dictionary
            key(1): assessment_id: Integer
            key(2): model_id: Integer
            key(3): rating: Integer
        :return:boolean
        """
        return self._db_register_feedback(self.feedback_type, dict_info)

    def retrieve_feedback_by_ids(self, dict_condition=None):
        """

        :param dict_condition:
            key(1) id: Integer
            key(2) assessment_id: Integer
            key(3) model_id: Integer
        :return:
        """
        return self._db_retrieve_feedback_by_ids(self.feedback_type, dict_condition)

    def retrieve_feedback_by_rating(self, dict_condition):
        """

        :param dict_condition: {"min_rating"  : Integer, "max_rating" : Integer, "sorting_order" : ASC or DESC}
        :return: List of Dictionaries
        """

        # Default Value
        min_rating = 0
        max_rating = 5
        sorting_order = "DESC"

        for i in list(dict_condition):
            if i is 'min_rating':
                min_rating = dict_condition[i]
            elif i is 'max_rating':
                max_rating = dict_condition[i]
            elif i is 'sorting_order':
                sorting_order = dict_condition[i]

        return self._db_retrieve_feedback_by_rating(self.feedback_type, min_rating=min_rating,
                                                      max_rating=max_rating, sorting_order=sorting_order)

    def update_feedback(self, id, dict_info):
        """

        :param id: Integer: feedback_id
        :param dict_info: any attributes want to change (feedback_type, as_id, m_id, rating, rated_date)
        :return:
        """
        return self._db_update_feedback(self.feedback_type, id, dict_info)

    def make_dummy_feedback(self, count=20):
        return self._db_make_dummy_feedback(self.feedback_type, count)

    def truncate_feedback(self):
        return self._db_truncate_feedback(feedback_type=self.feedback_type)


if __name__ == '__main__':
    af = AgingFeedback()
    ef = EmotionFeedback()
    ar = RecommendationFeedback()

    # af.register_feedback({"assessment_id": 30, "model_id": 1, "rating": 5})
    # ef.register_feedback({"assessment_id": 50, "model_id": 2, "rating": 3})
    # ar.register_feedback({"assessment_id": 70, "model_id": 3, "rating": 1})
    #
    # af.retrieve_feedback_by_ids()
    # ef.retrieve_feedback_by_ids()
    # ar.retrieve_feedback_by_ids()

    # for i in range(10000):
    # print(i)
    # print(af.retrieve_feedback_by_ids({"model_id": 102}))
    # ef.retrieve_feedback_by_ids({'id': 5, "assessment_id": 50})
    # ef.retrieve_feedback_by_ids({'id': 5})
    # ef.retrieve_feedback_by_ids({'id': 5, "model_id": 50})
    # ef.retrieve_feedback_by_ids({'assessment_id': 5, "model_id": 50})


    # ar.retrieve_feedback_by_ids()

    ef.update_feedback(3, {'as_id': '10'})

    # ar.retrieve_feedback_by_rating({'min_rating': 1})
    # af.retrieve_feedback_by_rating({"min_rating": 6})
    # ef.retrieve_feedback_by_rating({"max_rating": 5})
    # ar.retrieve_feedback_by_rating({"min_rating": 2})

    # af.update_feedback(65, {'rating': '3'})

    # af.make_dummy_feedback()
    # ef.make_dummy_feedback()
    # ar.make_dummy_feedback()

    # af.truncate_feedback()
    # ef.truncate_feedback()
    # ar.truncate_feedback()
