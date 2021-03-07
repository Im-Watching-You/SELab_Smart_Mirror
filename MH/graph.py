import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


data = pd.read_csv('./data, FGN, 0730.csv')
data = data.iloc[:,1:]

# data.plot.scatter(x="F5", y="F8")
# data.plot.scatter(x="F5", y="F9")
# data.plot.scatter(x="F5", y="F4")
# data.plot.scatter(x="F5", y="F1")
# data.plot.scatter(x="F8", y="F9")
# data.plot.scatter(x="F8", y="F4")
# data.plot.scatter(x="F8", y="F1")
# data.plot.scatter(x="F4", y="F1")
data.plot.scatter(x="Age", y="F5")
data.plot.scatter(x="Age", y="F8")
data.plot.scatter(x="Age", y="F6")
data.plot.scatter(x="Age", y="F9")
# data.plot.scatter(x="Age", y="F5")
# data.plot.scatter(x="Age", y="F6")
# data.plot.scatter(x="Age", y="F7")
# data.plot.scatter(x="Age", y="F8")
# data.plot.scatter(x="Age", y="F9")
# data.plot.scatter(x="Age", y="F10")
# data.plot.scatter(x="Age", y="F11")
# data.plot.scatter(x="Age", y="F12")
plt.show()

fig = plt.figure()
#
# ax = fig.add_subplot(111, projection='3d')
# z = data.loc[:, 'Age']
# x = data.loc[:, 'F9']
# y = data.loc[:, 'F8']
# ax.scatter(xs=x, ys=y, zs=z, zdir='z', s=20)
# plt.show()
#
# fig = plt.figure()
#
# ax = fig.add_subplot(111, projection='3d')
# z = data.loc[:, 'Age']
# x = data.loc[:, 'F1']
# y = data.loc[:, 'F4']
# ax.scatter(xs=x, ys=y, zs=z, zdir='z', s=20)
# plt.show()