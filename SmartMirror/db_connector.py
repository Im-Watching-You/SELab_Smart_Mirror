import pymysql.cursors
from datetime import datetime

class User:
    def __init__(self):
        self.conn = pymysql.connect(host='localhost', user='root', password='root', db='Smart_Mirror_System')
        self.info = dict(id=None, first_name=None, last_name=None, email=None, age=None,
                         phone_number=None)

    def close(self):
        self.conn.close()

    def create(self, user_info):
        self.update_info(user_info)
        self.insert_db()

    def update_info(self, info):
        #print(info)
        self.info.update(info)

    def insert_db(self):
        user_info = self.info
        with self.conn.cursor() as cursor:
            sql = 'INSERT INTO users (first_name, last_name, phone_number, email) VALUES (%s, %s, %s, %s)'
            cursor.execute(sql, (user_info['first_name'], user_info['last_name'], user_info['phone_number'],
                                 user_info['email']))
        self.conn.commit()
        # print(cursor.lastrowid)
        # print('insert data finish')
        # 1 (last insert id)

    def update_db(self):
        user_id = self.info['id']
        email = self.info['email']
        phone_number = self.info['phone_number']
        with self.conn.cursor() as cursor:
            sql = 'UPDATE users SET email = %s, phone_number = %s WHERE id = %s'
            cursor.execute(sql, (email, phone_number, user_id))
        self.conn.commit()
        #print(cursor.rowcount)  # 1 (affected rows)
        #print('update data finish')

    def read_id(self):
        with self.conn.cursor() as cursor:
            sql = 'SELECT id FROM users WHERE email = %s'
            cursor.execute(sql, (self.info['email'],))
            result = cursor.fetchone()
            #print(result)
            #print('select data finish')
            return result[0]


    def read_first_name(self):
        return self.info['first_name']

    def read_last_name(self):
        return self.info['last_name']

    def read_email(self):
        return self.info['email']

    def read_age(self):
        return self.info['age']

    def read_phone_number(self):
        return self.info['phone_number']


class UserSession:
    def __init__(self):
        self.conn = pymysql.connect(host='localhost', user='root', password='root', db='Smart_Mirror_System')

    def create(self,  user_id, emotion_id, emotion_value, age_value, pixel):
        time_value = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        with self.conn.cursor() as cursor:
            sql = 'INSERT INTO assessment (id, session_time, emotion_id, emotion, age, pixel) VALUES (%s, %s, %s, %s, %s, %s)'
            cursor.execute(sql, (user_id, time_value, emotion_id, emotion_value, age_value, pixel))
            # self.session.update(dict(id=user_id, session_time=time_value, emotion=emotion_value, age=age_value))
        self.conn.commit()
        self.conn.close()

    def update(self, num, user_id=None, emotion_id=None, emotion_value=None, age_value=None, pixel=None):
        with self.conn.cursor() as cursor:
            if emotion_id is not None:
                cursor.execute('UPDATE assessment SET emotion_id = %s  WHERE num = %s', (emotion_id, num))
            if emotion_value is not None:
                cursor.execute('UPDATE assessment SET emotion = %s  WHERE num = %s', (emotion_value, num))
            if age_value is not None:
                cursor.execute('UPDATE assessment SET age = %s  WHERE num = %s', (age_value, num))
            if user_id is not None:
                cursor.execute('UPDATE assessment SET id = %s  WHERE num = %s', (user_id, num))
            if pixel is not None:
                cursor.execute('UPDATE assessment SET pixel = %s  WHERE num = %s', (pixel, num))
        self.conn.commit()
        #print(cursor.rowcount)  # 1 (affected rows)
        #print('update data finish')
        self.conn.close()

    def delete(self):
        pass

    def read_num(self, nu):
        with self.conn.cursor() as cursor:
            sql = 'SELECT * FROM assessment WHERE first_name = %s'
            cursor.execute(sql, ())

    def read_id(self):
        pass

    def read_curr_emotion(self):
        pass

    def read_curr_age(self):
        pass


class UserProfile:
    list_user = []
    def __init__(self):
        self.conn = pymysql.connect(host='localhost', user='root', password='root', db='Smart_Mirror_System')

    def close(self):
        self.conn.close()

    def search_db(self, user_id):
        with self.conn.cursor() as cursor:
            sql = 'SELECT * FROM users WHERE id = %s'
            cursor.execute(sql, (user_id,))
            result = cursor.fetchone()
            #print(result)
            #print('select data finish')
            user_info = dict(id=result[0], first_name=result[1], last_name=result[2], phone_number=result[3],
                             email=result[4])
            user = User()
            user.update_info(user_info)
            return user

    def delete_db(self, user_id):
        with self.conn.cursor() as cursor:
            sql = 'DELETE FROM users WHERE id = %s'
            cursor.execute(sql, (user_id,))
        self.conn.commit()
        #print(cursor.rowcount)  # 1 (affected rows)
        #print('delete data finish')

    def load_user_list_db(self):
        user_list = []
        with self.conn.cursor() as cursor:
            sql = 'SELECT * FROM users'
            cursor.execute(sql)
            result = cursor.fetchall()
            for row in result:
                info = dict(id=row[0], first_name=row[1], last_name=row[2],
                            email=row[4], phone_number=row[3])
                #print(row)
                user = User()
                user.update_info(info)
                user_list.append(user)
            self.list_user = user_list
            return user_list


class DiarySession:
    def insert_diary_session(self, id, data):
        time_value = data['session_time']
        file_name = data['file_name']
        content = data['content']
        emotion_value = data['emotion']
        self.conn = pymysql.connect(host='localhost', user='root', password='root', db='Smart_Mirror_System')
        print(">>> insert_diary_session")
        with self.conn.cursor() as cursor:
            if len(emotion_value) == 1:
                sql = 'INSERT INTO diary (id, session_time, file_name, content, emotion1, value1) VALUES (%s, %s, %s, %s, %s, %s)'
                cursor.execute(sql, (
                    id, time_value, file_name, content, emotion_value[0]['tone_name'], emotion_value[0]['score']))
            elif len(emotion_value) == 2:
                sql = 'INSERT INTO diary (id, session_time, file_name, content, emotion1, value1, emotion2, value2) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)'
                cursor.execute(sql, (id, time_value, file_name, content, emotion_value[0]['tone_name'], emotion_value[0]['score'], emotion_value[1]['tone_name'], emotion_value[1]['score']))
            elif len(emotion_value) == 3:
                sql = 'INSERT INTO diary (id, session_time, file_name, content, emotion1, value1, emotion2, value2, emotion3, value3) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)'
                cursor.execute(sql, (
                id, time_value, file_name, content, emotion_value[0]['tone_name'], emotion_value[0]['score'],
                emotion_value[1]['tone_name'], emotion_value[1]['score'], emotion_value[2]['tone_name'], emotion_value[2]['score']))

            self.conn.commit()
            self.conn.close()

    def load_diary_session(self, id=None):
        self.conn = pymysql.connect(host='localhost', user='root', password='root', db='Smart_Mirror_System')
        print(">>> load_diary_session")
        session_list = []
        with self.conn.cursor() as cursor:
            if id is None:
                sql = 'SELECT * FROM diary ORDER BY num DESC limit 5'
            else:
                sql = 'SELECT * FROM diary ORDER BY num DESC limit 5'
            cursor.execute(sql)
            result = cursor.fetchall()
            for row in result:
                info = dict(id=row[1], session_time=row[2], file_name=row[3], content=row[4], emotion1=row[5], value1=row[6], emotion2=row[7], value2=row[8], emotion3=row[9], value3=row[10] )
                session_list.append(info)
            self.conn.close()
            return session_list

    def delet_diary_session(self, session_time):
        self.conn = pymysql.connect(host='localhost', user='root', password='root', db='Smart_Mirror_System')
        with self.conn.cursor() as cursor:
            sql = 'DELETE FROM diary WHERE session_time = %s'
            cursor.execute(sql, (session_time,))
            self.conn.commit()
            self.conn.close()
        print('delete_ data')


class UserSessionList:
    def __init__(self):
        # self.conn = pymysql.connect(host='localhost', user='root', password='root', db='Smart_Mirror_System')
        pass

    def load_session_list_db(self, id=None):
        self.conn = pymysql.connect(host='localhost', user='root', password='root', db='Smart_Mirror_System')
        session_list = []
        print(">>> load_session_list_db")
        session_list = []
        with self.conn.cursor() as cursor:
            if id is None:
                sql = 'SELECT * FROM assessment'
            else:
                sql = 'SELECT * FROM assessment WHERE id ='+str(id)
            cursor.execute(sql)
            result = cursor.fetchall()
            for row in result:
                info = dict(id=row[1], session_time=row[2], emotion_id=row[3], emotion=row[4], age=row[5])
                session_list.append(info)
                # print(info)

            self.conn.close()
            return session_list


    def close(self):
        if self.conn.open:
            self.conn.close()


if __name__ == '__main__':
    up = UserProfile()
    l = up.load_user_list_db()
    print(l)
    for u in up.list_user:
        print(u.read_first_name())
    print()
