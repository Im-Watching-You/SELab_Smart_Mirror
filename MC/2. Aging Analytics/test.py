l1 = [[1, 2, 3]]
l2 = [[4, 5, 6]]
l3 = l1+l2
print(l3)

q = [l3]
print(q)












# Make new_data.csv
# import pandas as pd
#
# df_mh = pd.read_csv("./data/data,0717b.csv")
# df_mh = df_mh.iloc[:, 3:]
# df_mh_path = df_mh.iloc[:, -1]
# print(df_mh.head())
# df_mc = pd.read_csv("./data/data4_4.csv")
# print(df_mc.head())
#
# # df_new = df_mc.append(df_mh, ignore_index=True, sort=False)
# # df_new = pd.concat([df_mc, df_mh], axis=1, join='inner')
# df_new = pd.merge(df_mc, df_mh, on='path')
# # df_new.to_csv("./data/new_data2.csv", mode='a')
#
# print(df_new.head())


# # Mean Value of Wrinkles
# def calculate_wrinkle_value(self, wrinkles):
#     sum = 0
#     count = 0
#     for f in wrinkles[0]:
#         sum += f
#         count += 1
#     result = float(sum) / float(count)
#     return result


# mean = self.calculate_wrinkle_value(features)
# 'Wrinkle_Features_Mean': mean

# wrinkle_value = result['Wrinkle_Features_Mean']
# print('\nAppearance Age:', age)
# print('Wrinkles: {:.2f}%'.format(wrinkle_value * 100.0))

