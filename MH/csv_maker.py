import csv
from random import *
import datetime

"""Aging Factor Maker """
# # f = open('./output.csv', 'w', encoding='utf-8', newline='')
# # wr = csv.writer(f)
# # date = [datetime.datetime(2019,7,1),datetime.datetime(2019,7,2),datetime.datetime(2019,7,3),datetime.datetime(2019,7,4),
# #         datetime.datetime(2019, 7, 5),datetime.datetime(2019,7,6),datetime.datetime(2019,7,7)]
# #
# # wr.writerow(
# #     ["id","u_id","date", "f1", "f2", "f3", "f4", "f5", "Age"])
# # for i in range(0, 2100):
#
#     wr.writerow([i+1, randint(1,10), date[i//300],randint(1, 100), randint(1, 100), randint(1, 100), randint(1, 100), randint(1, 100), randint(1, 100)])
# f.close()

"""Aging Factor Maker """
f = open('./output_emotion.csv', 'w', encoding='utf-8', newline='')
wr = csv.writer(f)
date = [datetime.datetime(2019,7,1),datetime.datetime(2019,7,2),datetime.datetime(2019,7,3),datetime.datetime(2019,7,4),
        datetime.datetime(2019, 7, 5),datetime.datetime(2019,7,6),datetime.datetime(2019,7,7)]
emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

wr.writerow(
    ["id","u_id","date", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "Emotion"])
for i in range(0, 2100):

    wr.writerow([i+1, randint(1,10), date[i//300],randint(1, 100), randint(1, 100), randint(1, 100), randint(1, 100),
                 randint(1, 100), randint(1, 100), randint(1, 100), randint(1, 100), randint(1, 100),
                 randint(1, 100), randint(0, 6)])
f.close()