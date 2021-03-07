from datetime import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import re

import matplotlib.pyplot as plt
import pandas as pd

test_date = "19940915"
# # print(datetime.strptime(test_date, "%Y%m%d"))
# # print(type(datetime.strptime(test_date, "%Y%m%d")))
#
# age = int((datetime.now() - datetime.strptime(test_date, "%Y%m%d")).days / 365)
# print(age)

# check_email = re.compile('^[a-zA-Z0-9+-_.]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$')
# pn = '01051319925'
# print(len(pn))

# print('MALE'.lower() == 'male')

# check_email = re.compile('^[a-zA-Z0-9+-_.]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$')  # https://dojang.io/mod/page/view.php?id=2439
# print(check_email.match('wch940naver.com') != None)


# start_date = '19940915'
# end_date = '20190621'
# sql_date = 'WHERE joined_date BETWEEN ' + start_date + ' AND ' + end_date
#
# print(sql_date)
#
# sql_date = 'WHERE joined_date BETWEEN ' + start_date + ' AND ' + str(datetime.now().strftime("%Y%m%d"))
#
# print(sql_date)
#

# a = {'name': 'park', 'phone': '010'}
# b = {'name': 'Lee'}
# c = {'name': 'Kwon', 'phone': '020', 'gender':'male'}

# def test(profile):
#     sql = 'UPDATE user SET'
#     list(profile.keys()) # list -> string (including ,)
#
#     for i in range(len(profile)):
#         if i is not len(profile)-1:
#             print(list(profile.keys())[i] + " = " + list(profile.values())[i] + ",")
#         else:
#             print(list(profile.keys())[i] + " = " + list(profile.values())[i])
#
# # def list_to_stirng_with_comma(list)
# #     for i in range(len(list)):
#
# def key_equation_value(dict):
#     equation =""
#     for i in range(len(dict)):
#         if i is not len(dict)-1:
#             equation += list(dict.keys())[i] + " = " + list(dict.values())[i] + ", "
#         else:
#             equation += list(dict.keys())[i] + " = " + list(dict.values())[i]
#     return equation
#
# print(key_equation_value(a))

# print(list(a.values())[0])
#
# dict1 = {'start_joined': '20151212', 'end_joined': '20190626', 'order_by': 'male_fisrt'}
# dict2 = {'start_joined': '20151212', 'order_by': 'male_first'}
# dict3 = {'order_by': 'male_fisrt'}
# dict4 = {"maintain": 1}
#
# for i in dict4:
#     if i is 'maintain':
#         exec(f"{i} = {2}")
#
# print(maintain)
#
# for i in dict1:
#     if i is 'order_by':
#         dict1[i] = 'hey'
#
# print(dict1)

# str = 'aging'
#
# if str in {'aging','abcing','abding)'}:
#     print('right')
# result = {"id": 1}
# result.append({"test": 1})

# print(list(result)[0])

# model_type = "aging_model"
# model_type = model_type.split('_')[0]
# print(model_type)
#
# maintain, improve, prevent = 0, 0, 0
#
# print(maintain)

# name, user_id, phone_number, start_joined, end_joined, order_by = None, None, None, None, None, None
#
# print(type([]))

# result = ()
#
# if result is ():
#     print( [])
# else:
#     print (())

# id, t_id, s_id = ['None',]
#
# print(datetime.today().date()-timedelta(days=7))
# print(datetime.today() - relativedelta(months=1))

# start_date = (datetime.today() - relativedelta(months=1)).date() + timedelta(days=1)
# print(start_date)

# f1_sum, f2_sum, f3_sum, f4_sum, f5_sum, f6_sum, f7_sum, f8_sum, f9_sum, f10_sum = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

# print(datetime.now().date())
# start_date = datetime.today().date()-timedelta(days=6)
# start_date = datetime.today().date() - relativedelta(months=3)
#
# print(type(datetime.today().date()+timedelta(days=1)))
#
# list_dict = [{'f1':1,'f2',2}]
#
# print(list_dict.keys())
# print(type(datetime.now()))
#
# print(type(start_date))


# f1_start_sum, f2_start_sum, f3_start_sum, f4_start_sum, f5_start_sum, f6_start_sum, \
#     f7_start_sum, f8_start_sum, f9_start_sum, f10_start_sum = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
#
# print(f1_start_sum)

# dict_start_sum = {'f1': 10}
# dict_start_sum['f1'] += 10
# print(dict_start_sum['f1'])
#
# dict_end_sum = {'f1': 20}
#
# factor_name = 'f1'
# factor_rate = dict_end_sum[f'{factor_name}'] / dict_start_sum[f'{factor_name}']
# print(factor_rate)

# factor_rate = 1.04
# if 1.05 >= factor_rate >= 0.95:
#     print('shit')
#
# result = []
# result.append({f"{factor_rate}":1})
# print(result)

# a = 2/7
# b = 3/5
# print(round(a/b, 2))

# a = plt.hist((0,0,0,0,), range=(-0.5, 6.5), histtype='bar', rwidth=0.5)
# print(type(a))
# print(a)
#

# dict_sum = {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0}
# #
# print(type(dict_sum.keys()))
# print('angry' in dict_sum.keys())
# for i in dict_sum:
#     print(dict_sum[i])


# condition_string = ""
#
# if condition_string is "":
#     print('ok')
#
# print(datetime.now() + timedelta(weeks=1))
# print(datetime.now() + timedelta(days=7))

#
# list_age = [10, 14, 15, 17, 10, 13, 12, 15]
#
#
# a = plt.hist(list_age, range=(min(list_age)-0.5, max(list_age)+0.5), histtype='bar', rwidth=0.5, bins= len(list_age))
# plt.ylabel('frequency')
#
#
# # TODO: delete test code [~:2]
# user_id = '11'
# duration = 'month'
#
# plt.title(f'User ID : {user_id} \'s {duration}ly Predicted Age Frequency')
#
# plt.show()

# string = '01051319925'
#
# if len(string) is not 11 :
#
#     print(len(string))
#     print(string[0:3])
#
#     print('test')
#
# else:
#     print('wow')
#
# if len(string) != 11 or string[0:3] != '010':
#
#     print(len(string))
#     print(string[0:3])
#
#     print('test')
#
# else:
#     print('wow')
#
# print('string : ', id(string))
# print('string[0:3] : ', id(string[0:3]))
# print('\'010\' : ', id('010'))

# print(id(type('abc')))
# print(id(type('dbc')))

list_age = [10, 12, 13, 10, 22, 44, 55]

dict_sum = dict()

for i in list_age:
    if str(i) in dict_sum.keys():
        dict_sum[str(i)] += 1
    else:
        dict_sum[str(i)] = 1

print(dict_sum)
ratio = [round(dict_sum[age] / 7 * 100, 1)for age in dict_sum]

print(ratio)

test = [i for i in dict_sum]

print(test)