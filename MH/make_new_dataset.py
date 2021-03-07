import pandas as pd

df_mh = pd.read_csv("./data,0717b.csv")
df_mh = df_mh.iloc[:, 2:]
print(df_mh.head())
df_mc = pd.read_csv("./data3.csv")
df_mc = df_mc.iloc[:, 1:]
print(df_mc.head())

# df_new = df_mc.append(df_mh, ignore_index=True, sort=False)
# df_new = pd.concat([df_mc, df_mh], axis=1, join='inner')
df_new = pd.merge(df_mc, df_mh, on='path')
df_new.to_csv("./new_data.csv", mode='a')

print(df_new.head())




