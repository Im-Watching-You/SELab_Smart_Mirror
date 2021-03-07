import WC.dbconnector as db
import random
from datetime import datetime


class Feedback:
    def __init__(self):
        self.coon = None

    def _db_register_feedback(self, feedback_type, dict_info):

        target = feedback_type

        if target is 'aging_feedback':
            value_name = 'ad_id'

        elif target is 'emotion_feedback':
            value_name = 'ed_id'

        elif target is 'recommendation_feedback':
            value_name = 'ar_id'

        for i in list(dict_info):
            if i in {'aging_diagnosis_id', 'emotion_diagnosis_id', 'aging_recommendation_id'}:
                assess_id = dict_info[i]
            elif i is 'model_id':
                m_id = dict_info[i]
            elif i is 'rating':
                rating = dict_info[i]

        sql = f"INSERT INTO {target} ({value_name}, m_id, rating, rated_date)" \
            f"VALUES ({assess_id}, {m_id}, {rating}, \'{datetime.now()}\')"

        db.connect_to_db(self)
        with self.conn.cursor() as cursor:
            try:
                cursor.execute(sql)
                print(sql)
                db.disconnect_from_db(self)
                return True

            except Exception as e:
                print(e)
                db.disconnect_from_db(self)
                return False

    def _db_retrieve_feedback_by_ids(self, feedback_type, id=None, assess_id=None, model_id=None):

        if feedback_type is 'aging_feedback':
            value_name = 'ad_id'

        elif feedback_type is 'emotion_feedback':
            value_name = 'ed_id'

        elif feedback_type is 'recommendation_feedback':
            value_name = 'ar_id'

        sql = f"SELECT * FROM {feedback_type} "

        if id is None and assess_id is None and model_id is None:
            condition = ""

        elif id is not None:
            condition = f"WHERE id = \'{id}\'"

        elif assess_id is not None and model_id is not None:
            condition = f"WHERE {value_name} = \'{assess_id}\' and m_id = \'{model_id}\'"

        elif assess_id is not None and model_id is None:
            condition = f"WHERE {value_name} = \'{assess_id}\'"

        elif assess_id is None and model_id is not None:
            condition = f"WHERE m_id = \'{model_id}\'"

        db.connect_to_db(self)
        with self.conn.cursor() as cursor:
            try:
                cursor.execute(sql+condition)
                result = cursor.fetchall()
                db.disconnect_from_db(self)
                if result is ():
                    return []
                else:
                    return result

            except Exception as e:
                print(e)
                print("_db_retrieve_feedback_by_ids : error")
                db.disconnect_from_db(self)
                return []

    def _db_retrieve_feedback_by_rating(self, feedback_type, min_rating=None, max_rating=None, sorting_order=None):

        sql = f"SELECT * FROM {feedback_type} WHERE rating >= {min_rating} and rating <= {max_rating} ORDER BY \'{sorting_order}\'"
        db.connect_to_db(self)
        with self.conn.cursor() as cursor:
            try:
                cursor.execute(sql)
                result = cursor.fetchall()
                db.disconnect_from_db(self)
                return result

            except Exception as e:
                print(e)
                print("_db_retrieve_feedback_by_rating : error" )
                db.disconnect_from_db(self)
                return []

    def _db_retrieve_feedback_by_recorded_date(self, feedback_type, start_date='20000101', end_date='20991231'):

        sql = f"SELECT * FROM {feedback_type} WHERE recorded_date >= \'{start_date}\' and recorded_date <= \'{end_date}\'"

        db.connect_to_db(self)
        with self.conn.cursor() as cursor:
            try:
                cursor.execute(sql)
                result = cursor.fetchall()
                db.disconnect_from_db(self)
                return result

            except Exception as e:
                print(e)
                print("_db_retrieve_feedback_by_recorded_date : error" )
                db.disconnect_from_db(self)
                return []

    def _db_update_feedback(self, feedback_type, id=None, rating=None):
        """

        :param feedback_type: type of feedback in ("aging_feedback", "emotion_feedback", "recommendation_feedback")
        :param id: Integer
        :param rating: Integer
        :return:
        """

        sql = f"UPDATE {feedback_type} SET rating = {rating} WHERE id = {id}"

        db.connect_to_db(self)
        with self.conn.cursor() as cursor:
            try:
                cursor.execute(sql)
                db.disconnect_from_db(self)
                return True

            except Exception as e:
                print(e)
                print("_db_update_feedback : error")
                db.disconnect_from_db(self)
                return False

    def _db_truncate_feedback(self, feedback_type):

        option = input('<STAFF ONLY> This method will delete entire rows of the table. Are you Sure? (Y/N)')

        if option.lower() == 'y':
            db.connect_to_db(self)
            with self.conn.cursor() as cursor:
                try:
                    sql = f"TRUNCATE TABLE smart_mirror_system.{feedback_type}"
                    cursor.execute(sql)
                    db.disconnect_from_db(self)
                    return True

                except Exception as e:
                    print(e)
                    db.disconnect_from_db(self)
                    return False
        else:
            print('Truncate Canceled.')
            return False


class AgingFeedback(Feedback):

    feedback_type = 'aging_feedback'

    def register_feedback(self, dict_info):
        """

        :param dict_info: {"aging_diagnosis_id" : Integer , "model_id" : Integer, "rating" : Integer}
        :return: Boolean
        """
        return self._db_register_feedback(self.feedback_type, dict_info)

    def retrieve_feedback_by_ids(self, dict_condition=None):

        if dict_condition is None:
            return self._db_retrieve_feedback_by_ids(feedback_type=self.feedback_type)

        id = None
        ad_id = None
        m_id = None

        for i in list(dict_condition):
            if i is 'id':
                id = dict_condition[i]
            elif i is 'ad_id':
                ad_id = dict_condition[i]
            elif i is 'm_id':
                m_id = dict_condition[i]

        return self._db_retrieve_feedback_by_ids(feedback_type=self.feedback_type, id=id,
                                                   assess_id=ad_id, model_id=m_id)

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

    def update_feedback(self, dict_info):
        # db method are implemented
        pass

    def make_dummy_feedback(self, count=100):
        db.connect_to_db(self)
        with self.conn.cursor() as cursor:

                try:
                    for i in range(count):
                        sql = f"INSERT INTO aging_feedback (ad_id, m_id, rating, rated_date ) VALUES(\'{random.randrange(1,100)}\', " \
                            f" \'{random.randrange(1,100)}\', {random.randrange(1,5)}, \'{datetime.now()}\')"
                        cursor.execute(sql)

                except Exception as e:
                    print(e)
                    db.disconnect_from_db(self)
                    return False

                db.disconnect_from_db(self)
                return True

    def truncate_feedback(self):
        return self._db_truncate_feedback(feedback_type=self.feedback_type)


class EmotionFeedback(Feedback):

    feedback_type = 'emotion_feedback'

    def register_feedback(self, dict_info):
        """

        :param dict_info: {"emotion_diagnosis_id" : Integer , "model_id" : Integer, "rating" : Integer}
        :return: Boolean
        """
        return self._db_register_feedback(self.feedback_type, dict_info)

    def retrieve_feedback_by_ids(self, dict_condition=None):

        if dict_condition is None:
            return self._db_retrieve_feedback_by_ids(feedback_type=self.feedback_type)

        id = None
        ed_id = None
        m_id = None

        for i in list(dict_condition):
            if i is 'id':
                id = dict_condition[i]
            elif i is 'ed_id':
                ed_id = dict_condition[i]
            elif i is 'm_id':
                m_id = dict_condition[i]

        return self._db_retrieve_feedback_by_ids(feedback_type=self.feedback_type, id=id, assess_id=ed_id, model_id=m_id)

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

    def update_feedback(self, dict_info):
        # db method are implemented
        pass

    def make_dummy_feedback(self, count=100):

        db.connect_to_db(self)
        with self.conn.cursor() as cursor:

                try:
                    for i in range(count):
                        sql = f"INSERT INTO emotion_feedback (ed_id, m_id, rating, rated_date ) VALUES(\'{random.randrange(1, 100)}\', " \
                            f" \'{random.randrange(1, 100)}\', {random.randrange(1, 5)}, \'{datetime.now()}\')"
                        cursor.execute(sql)
                        print(sql)

                except Exception as e:
                    print(e)
                    db.disconnect_from_db(self)
                    return False

                db.disconnect_from_db(self)
                return True

    def truncate_feedback(self):
        return self._db_truncate_feedback(feedback_type=self.feedback_type)


class RecommendationFeedback(Feedback):

    feedback_type = 'recommendation_feedback'

    def register_feedback(self, dict_info):
        """

        :param dict_info: {"aging_recommendation_id" : Integer , "model_id" : Integer, "rating" : Integer}
        :return: Boolean
        """
        return self._db_register_feedback(self.feedback_type, dict_info)

    def retrieve_feedback_by_ids(self, dict_condition=None):

        if dict_condition is None:
            return self._db_retrieve_feedback_by_ids(feedback_type=self.feedback_type)

        id = None
        ar_id = None
        m_id = None

        for i in list(dict_condition):
            if i is 'id':
                id = dict_condition[i]
            elif i is 'ar_id':
                ar_id = dict_condition[i]
            elif i is 'm_id':
                m_id = dict_condition[i]

        return self._db_retrieve_feedback_by_ids(feedback_type=self.feedback_type, id=id,
                                                   assess_id=ar_id, model_id=m_id)

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

    def update_feedback(self, dict_info):
        # db method are implemented
        pass

    def make_dummy_feedback(self, count=100):
        db.connect_to_db(self)
        with self.conn.cursor() as cursor:

                try:
                    for i in range(count):
                        sql = f"INSERT INTO recommendation_feedback (ar_id, m_id, rating, rated_date ) VALUES(\'{random.randrange(1, 100)}\', " \
                            f" \'{random.randrange(1, 100)}\', {random.randrange(1, 5)}, \'{datetime.now()}\')"
                        cursor.execute(sql)
                        print(sql)

                except Exception as e:
                    print(e)
                    db.disconnect_from_db(self)
                    return False

                db.disconnect_from_db(self)
                return True

    def truncate_feedback(self):
        return self._db_truncate_feedback(feedback_type=self.feedback_type)


# af = AgingFeedback()
# ef = EmotionFeedback()
# ar = RecommendationFeedback()
#
# print(af.retrieve_feedback_by_rating({"min_rating": 6}))
# print(ef.retrieve_feedback_by_rating({"max_rating": 5}))
# print(ar.retrieve_feedback_by_rating({"min_rating": 2}))

# af.register_feedback({"aging_diagnosis_id": 30, "model_id": 1, "rating": 5})
# ef.register_feedback({"emotion_diagnosis_id": 30, "model_id": 1, "rating": 5})
# ar.register_feedback({"aging_recommendation_id": 30, "model_id": 1, "rating": 5})
#
# af.retrieve_feedback_by_ids()
# ef.retrieve_feedback_by_ids()
# ar.retrieve_feedback_by_ids()
#
# af.retrieve_feedback_by_ids({"model_id": 1})
# ef.retrieve_feedback_by_ids({"model_id": 1})
# ar.retrieve_feedback_by_ids({"model_id": 1})
#
# af.make_dummy_feedback()
# ef.make_dummy_feedback()
# ar.make_dummy_feedback()

# af.truncate_feedback()
# ef.truncate_feedback()
# ar.truncate_feedback()
