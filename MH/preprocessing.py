import pandas as pd
import numpy as np

data = pd.read_csv("./data,0724.csv")
data = data.iloc[:,1:]
print()
df_ages = data.loc[:, 'Age'].tolist()
list_age = []
[list_age.append(v) for v in df_ages if v not in list_age]
d = sorted(list_age)
print(type(list_age), d)
low = 0.45
high = 0.55

key_factor = data.columns.values[1:13]
c = 0
for age in d:
    c += 1
    data_curr = data.loc[data['Age'] == age]
    for i in data_curr.index.tolist():
        count = 0
        for k in key_factor:
            d_cur_attr = data_curr.loc[:, k]
            qd = d_cur_attr.quantile([low, high]).tolist()
            if qd[0] <= data_curr.loc[i, k] <= qd[1]:
                count += 1
        if count >= 3:
            data = data.drop(i, axis=0)
    print(round(c/len(list_age)*100), "%")
    # quant_df = data_curr.quantile()
print(len(data))
print(data)
data.to_csv("./data_new, 0730c.csv", mode='a')



