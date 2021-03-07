import WC.person as ps
import re
import random
from datetime import datetime
import smtplib
from email.mime.text import MIMEText


class ProfileManager:

    @staticmethod
    def _is_profile_valid(dict_profile):

        # key order: password(0), gender(1), birth_date(2), first_name(3), last_name(4), phone_number(5), email(6), ap_id(or tier)(7)

        if len(dict_profile) is 8:

            # Integrity Check : password
            # condition : max(len) = 16
            if len(list(dict_profile.values())[0]) > 16:
                print('Maximum length of Password should be under 16 number of characters.')
                return False

            # Integrity Check : gender
            # condition : only male or female
            if list(dict_profile.values())[1].lower() != 'male' and list(dict_profile.values())[1].lower() != 'female':
                print('Invalid Gender. Please insert \'male\'or \'female\' regardless of Case.')
                return False

            # Integrity Check : birth_date
            # condition : '19940915' format
            # When Birth Date exceeds today : The man who came from the future
            try:
                if len(list(dict_profile.values())[2]) > 8:
                    print('ERROR ::: Birth Date Format ex) 19990101')
                    return False
                if (datetime.now() - datetime.strptime(list(dict_profile.values())[2], "%Y%m%d")).days < 0:
                    # https://brownbears.tistory.com/18
                    print('Invalid Birth Date.')
                    return False
            except ValueError:
                print()

            # Integrity Check : first_name, last_name
            # condition : name with only alphabets.
            check_name = re.compile('^[a-zA-z]+$')
            if check_name.match(list(dict_profile.values())[3]) is None:
                print(f'Invalid First Name. \'{list(dict_profile.values())[3]}\'')
                return False
            if check_name.match(list(dict_profile.values())[4]) is None:
                print(f'Invalid Second Name. \'{list(dict_profile.values())[4]}\'')
                return False

            # Integrity Check : phone_number
            # condition : '01051319925' format
            if list(dict_profile.values())[5][0:3] != '010' or len(list(dict_profile.values())[5]) is not 11:
                print('Invalid Phone Number')
                return False

            # Integrity Check : email
            # condition : 'ABC@BBB.com' format
            check_email = re.compile('^[a-zA-Z0-9+-_.]+@[a-zA-Z0-9-]+.[a-zA-Z0-9-.]+$')
            # https://dojang.io/mod/page/view.php?id=2439
            if check_email.match(list(dict_profile.values())[6]) is None:
                print('Invalid Email Address')
                return False

            return True

        # At least one parameter is missing
        else:
            print('Empty value detected')
            return False


class UserManager(ProfileManager):

    def __init__(self):
        self.user = ps.User()

    def register_user(self, dict_profile):

        # If validity function returns True
        if self._is_profile_valid(dict_profile):

            return self.user.register_user(dict_profile)

        # If the function returns Fals
        else:
            print("\nPlease Try Again.")
            return False

    def deregister_user(self, user_id):
        return self.person.deregister_user(user_id)

    def retrieve_user(self, dict_condition=None):

        if dict_condition:
            print(dict_condition, list(dict_condition))
            for i in list(dict_condition):
                if i is 'phone_number':
                    if dict_condition[i][0:3] != '010' or len(dict_condition[i]) != 11:
                        print('Invalid Phone Number', dict_condition[i][0:3], dict_condition[i][0:3] != "010")
                        return []
                elif i is 'start_joined':
                    if (datetime.now() - datetime.strptime(dict_condition[i], "%Y%m%d")).days < 0:
                        print('Invalid start joined date')
                        return []
                elif i is 'end_joined':
                    if (datetime.strptime(dict_condition[i], "%Y%m%d") - datetime.strptime('20000101', "%Y%m%d")).days < 0:
                        print('Invalid start joined date')
                        return []
                elif i is 'order_by':
                    if dict_condition[i] not in ['rand', 'male_first', 'female_first', 'birth_asc', 'birth_desc',
                                                    'joined_date_asc', 'joined_date_desc']:
                        print("Invalid ordering type")
                        return []

            return self.user.retrieve_user(dict_condition)

        else:
            return self.user.retrieve_user()

    def retrieve_user_by_age_gender(self, dict_condition):
        """
        :param dict_condition : Dictionary
            key(1) : age_from : String
            key(2) : age_to : String
            key(3) : gender
        :return: list of dictionaries

        """
        age_from = '0'
        age_to = '120'
        gender = None

        for i in dict_condition:
            if i is 'from':

                if 0 <= int(dict_condition[i]) <= 120:
                    age_from = dict_condition[i]

                else:
                    print("Invalid Age.")
                    return []

            elif i is 'to':
                if 0 <= int(dict_condition[i]) <= 120:
                    age_to = dict_condition[i]

                else:
                    print("Invalid Age.")
                    return []

            elif i is 'gender':
                dict_condition[i] = dict_condition[i].lower()
                if dict_condition[i] not in ['male', 'female']:
                    print('Invalid Gender.')
                    return []

        return self.user.retrieve_user_by_age_gender(dict_condition)

    def update_user_profile(self, user_id, dict_info=None):
        """

        :param user_id: Integer
        :param dict_info:
            key(0): 'password' : String
            kwy(1): 'phone_number : String
            key(2): 'email' : String
        :return: Boolean
        """

        for i in list(dict_info):
            if i is 'password':
                if len(dict_info[i]) > 16:
                    print('Maximum length of Password should be under 16 number of characters.')
                    return False

            elif i is 'phone_number':
                if dict_info[i][0:3] != '010' or len(dict_info[i]) is not 11:
                    print('Invalid Phone Number')
                    return False

            elif i is 'email':
                check_email = re.compile('^[a-zA-Z0-9+-_.]+@[a-zA-Z0-9-]+.[a-zA-Z0-9-.]+$')
                # https://dojang.io/mod/page/view.php?id=2439
                if check_email.match(dict_info[i]) is None:
                    print('Invalid Email Address')
                    return False

            elif i is 'gender':
                if dict_info[i].lower() not in ('male', 'female'):
                    print(f'Invalid gender : \'{dict_info[i]}\'')
                    return False

        return self.user.update_user_profile(user_id=user_id, dict_info=dict_info)

    def get_id_list(self):
        return self.user.get_id_list()

    # When user
    def update_new_password(self, dict_info):
        """
        :param dict_info:
            key(1): id
            key(2): email
        :return:boolean
        """
        id, email = None, None

        for i in list(dict_info):
            if i is 'id':
                id = dict_info[i]
            elif i is 'email':
                email = dict_info[i]

        if id and email:
            # To check given id of email is identical to given email
            if email == ps.User().retrieve_user({"user_id": id})[0]['email']:
                verification_code = random.randrange(10000000, 99999999)

                s = smtplib.SMTP('smtp.gmail.com', 587)
                s.starttls()
                s.login('wch940@gmail.com', 'qafp iwqs zuhu hukw')

                msg = MIMEText(f'verification code = {verification_code}'
                               f'\n Please Input this code')
                msg['Subject'] = '[Smart_Mirror_System] Verification Code'
                msg['To'] = f'{email}'
                s.sendmail('Smart_Mirror_System', email, msg.as_string())
                print(f"Verification Code has been sent with {email}")
                s.quit

                try:
                    user_code = int(input("Please input Verification Code sent to email.\n"
                                                  " Verification Code: "))
                except ValueError as v:
                    print("Input should be Integer. Please try again")
                    return False

                if verification_code == user_code:
                    new_password = input("Please input new password: ")
                    ps.User().update_user_profile(user_id=id, dict_info={'password': new_password})

                    return True
                else:
                    print("Incorrect Verification Code. Please try again.")
                    return False

            else:
                print("There is no email matching with given id. Please Check again.")
                return False
        else:
            print("Some input data is missing.")
            return False


class StaffManager(ProfileManager):

    def __init__(self):
        self.staff = ps.Staff()

    def register_staff(self, dict_profile):

        # If validity function returns True
        if self._is_profile_valid(dict_profile):

            return self.staff.register_staff(dict_profile)

        # If the function returns False
        else:
            print("\nPlease Try Again")
            return False

    def deregister_staff(self, staff_id):
        return self.staff.deregister_staff(staff_id)

    def retrieve_staff(self, dict_condition=None):

        if dict_condition:
            for i in list(dict_condition):
                if i is 'phone_number':
                    if dict_condition[i][0:3] is not '010' or len(dict_condition) is not 11:
                        print('Invalid Phone Number')
                        return False
                elif i is 'start_joined':
                    if (datetime.now() - datetime.strptime(dict_condition[i], "%Y%m%d")).days < 0:
                        print('Invalid start joined date')
                        return False
                elif i is 'end_joined':
                    if (datetime.strptime(dict_condition[i], "%Y%m%d") - datetime.strptime('20000101', "%Y%m%d")).days < 0:
                        print('Invalid start joined date')
                        return False
                elif i is 'order_by':
                    if dict_condition[i] not in ['rand', 'male_first', 'female_first', 'birth_asc', 'birth_desc',
                                                    'joined_date_asc', 'joined_date_desc']:
                        print("Invalid ordering type")
                        return False

            return self.staff.retrieve_staff(dict_condition)

        else:
            return self.staff.retrieve_staff()

    def update_staff_profile(self, staff_id, dict_info):
        """

        :param staff_id: Integer
        :param dict_info:
            key(0): 'password' : String
            kwy(1): 'phone_number : String
            key(2): 'email' : String
        :return: Boolean
        """

        for i in list(dict_info):
            if i is 'password':
                if len(dict_info[i]) > 16:
                    print('Maximum length of Password should be under 16 number of characters.')
                    return False

            elif i is 'phone_number':
                if dict_info[i][0:3] != '010' or len(dict_info[i]) is not 11:
                    print('Invalid Phone Number')
                    return False

            elif i is 'email':
                check_email = re.compile('^[a-zA-Z0-9+-_.]+@[a-zA-Z0-9-]+.[a-zA-Z0-9-.]+$')
                # https://dojang.io/mod/page/view.php?id=2439
                if check_email.match(dict_info[i]) is None:
                    print('Invalid Email Address')
                    return False

        return self.person.update_staff_profile(staff_id=staff_id, dict_info=dict_info)

@staticmethod
def um_test_code():
    """
    For Testing User Manger
    :return: None
    """
    a = {'password': 'testtesttesttest', 'gender': 'male', 'birth_date': '19940915', 'first_name': 'wonjong',
         'last_name': 'Lee', 'phone_number': '01099983338', 'email': 'test@test.com', 'ap_id': '1'}

    um = UserManager()

    um.register_user(a)

    um.deregister_user(286)

    """
    Testing retrieve_all_users()
    """
    um.retrieve_all_users({'end_joined': '20190624', 'order_by': 'male_first'})

    """
    Testing retrieve_user()
    """
    um.retrieve_user({'name': 'Rhys Lam', 'phone_number': '01051786262'})
    um.retrieve_user({'phone_number': '01051786262'})
    um.retrieve_user()

    """
    Testing update_user()
    """
    um.update_user()
    um.update_user({'user_id': '1', 'phone_number': '01012345678'})
    um.update_user({'user_id': '2', 'phone_number': '01012345674', 'email': 'test@testtest.com'})

@staticmethod
def sm_test_code():

    a = {'password': 'testtesttesttest', 'gender': 'male', 'birth_date': '19940915', 'first_name': 'wonjong', 'last_name': 'Lee',
             'phone_number': '01099983338', 'email': 'test@test.com', 'tier': '1'}

    b = {'password': 'testtesttesttest', 'gender': 'male', 'birth_date': '19940915', 'first_name': 'wonjong', 'last_name': 'Lee',
             'phone_number': '01099983338', 'email': 'test@test.com'}

    sm = StaffManager()

    """
    Testing register_staff()
    """
    # sm.register_staff() # error : any type of dictionary should be passed
    # sm.register_staff(a)
    # sm.register_staff(b)

    """
    Testing deregister_staff()
    """
    # sm.deregister_staff(1)
    # sm.deregister_staff(24) #error : there is no staff_id '24' but it doesn't raise exception. just 0 row effected .

    """
    Testing retrieve_all_staffs()
    """

    # sm.retrieve_all_staffs()
    # sm.retrieve_all_staffs({'end_joined': '20190303'})
    # sm.retrieve_all_staffs({'start_joined': '20190625', 'order_by': 'tier_asc'})
    # sm.retrieve_all_staffs({'start_joined': '20190627', 'end_joined': '20190628', 'order_by': 'tier_asc'}))
    # sm.retrieve_all_staffs({'start_joined': '20190627', 'end_joined': '20190627', 'order_by': 'tier_asc'})

    """
    Testing retrieve_staff()
    """

    # sm.retrieve_staff()
    # sm.retrieve_staff({'name': 'Joe Evans'})
    # sm.retrieve_staff({'name': 'JoeEvans'})
    # sm.retrieve_staff({'phone_number': '01011112222'})

    """
    Testing update_staff_profile
    """

    # sm.update_staff_profile({'staff_id': '2', 'email': 'haha@gmail.com'})


if __name__ == '__main__':
    um = UserManager()
    um.register_user({"password": 'test123', "gender": 'male', "birth_date": '19940915',
                        "first_name": 'gwanoong', 'last_name': 'Cheon', 'phone_number': '01031319876',
                        'email': 'test129987511@test.com', 'ap_id': 1})

    # um.update_user_profile(1, {'gender': 'testtest'})
    # um.update_new_password({'id':284,'email':'wch940@naver.com'})
    # print(um.retrieve_user_by_age_gender({'gender': 'made'}))

    sm = StaffManager()
    # sm.register_staff({"password": 'test123', "gender": 'male', "birth_date": '19940915',
    #                     "first_name": 'Woochan', 'last_name': 'Park', 'phone_number': '01051319876',
    #                     'email': 'test1299875@test.com', 'tier': 1})
    # print(sm.retrieve_staff({"first_nam":'male'}))
