import sklearn
import pandas as pd

data = pd.read_csv("./data.csv")
# print(type(data))
# print(data.head())
#
# data = data.head()
print(data)
d = data["Age"]
corr_matrix = data.corr(method='pearson')
print(corr_matrix['Age'].sort_values(ascending=False))
up_20 = data["Age"] > 20
down_50 = data["Age"] <= 50
data = data[up_20 & down_50]
data = data.iloc[:,1:]
print(data)
cor = data.corr()
print(cor['Age'].sort_values(ascending=False))

# data = data["Age"] <= 50
# print(len(data))
# age_dic = {'0-10':0,
#            '11-20':0,
#            '21-30':0,
#            '31-40':0,
#            '41-50':0,
#            '51-60':0,
#            '61-70':0,
#            '71-80':0,
#            '81-90':0,
#            '91-100':0,
#            }
# print(d)
# for i in d:
#     if i <=10:
#         age_dic['0-10']+=1
#     elif 10 < i <=20:
#         age_dic['11-20'] += 1
#     elif 20 < i <=30:
#         age_dic['21-30'] += 1
#     elif 30 < i <=40:
#         age_dic['31-40'] += 1
#     elif 40 < i <=50:
#         age_dic['41-50'] += 1
#     elif 50 < i <=60:
#         age_dic['51-60'] += 1
#     elif 60 < i <=70:
#         age_dic['61-70'] += 1
#     elif 70 < i <=80:
#         age_dic['71-80'] += 1
#     elif 80 < i <=90:
#         age_dic['81-90'] += 1
#     elif 90 < i <=100:
#         age_dic['91-100'] += 1
# # print(corr_matrix['Age'])
# print(age_dic)