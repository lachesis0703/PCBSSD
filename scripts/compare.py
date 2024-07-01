import csv
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams

df1 = pd.read_csv('syndet/scatters/l1loss.csv')  # csv文件所在路径
step1 = df1['Step'].values.tolist()
loss1 = df1['Value'].values.tolist()

df2 = pd.read_csv('syndet/scatters/l2loss.csv')
step2 = df2['Step'].values.tolist()
loss2 = df2['Value'].values.tolist()

df3 = pd.read_csv('syndet/scatters/smoothl1loss.csv')
step3 = df3['Step'].values.tolist()
loss3 = df3['Value'].values.tolist()

# 设置绘图风格，使用科学论文常见的线条样式和颜色
plt.style.use('seaborn-whitegrid')
# 设置字体和字号
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
# plt.ylim(0, 1)

plt.plot(step1, loss1, color='blue', label='L1 loss')
plt.plot(step2, loss2, color='green', label='L2 loss')
plt.plot(step3, loss3, color='red', label='Smooth L1 loss')
plt.xlabel('epochs')
plt.ylabel('loss')

plt.legend(fontsize=8)  # 图注的大小
plt.savefig('figure.eps')
plt.show()