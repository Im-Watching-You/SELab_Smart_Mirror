# DB Manage Module
# author  : Woo Chan Park
# version : 2019.06.18

import pymysql.cursors


# Configure DB setting
def connect_to_db(self):
    try:
        self.conn = pymysql.connect(host='203.253.23.45', user='root', password='root', db='Smart_Mirror_System',
                                    charset='utf8', autocommit=True, cursorclass=pymysql.cursors.DictCursor)
        return True

    except Exception as e:
        print(e)
        return False


def disconnect_from_db(self):
    self.conn.close()
