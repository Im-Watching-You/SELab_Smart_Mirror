import pymysql.cursors

# Configure DB setting
def connect_to_db(self):
    try:
        self.conn = pymysql.connect(host='203.253.23.27', user='root', password='root', db='Smart_Mirror_System',
                                    charset='utf8', autocommit=True, cursorclass=pymysql.cursors.DictCursor)
        return True
    except Exception:
        return False

def disconnect_from_db(self):
    self.conn.close()
    print("success?")
