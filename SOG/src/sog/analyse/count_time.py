import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
data = pd.read_csv('data/output/379_processing_times2 copy.csv')

# 计算时间的平均值
average_time = data['Time'].mean()
print(f"平均时间: {average_time}")

# 找到最大值和最小值
max_time = data['Time'].max()
min_time = data['Time'].min()
print(max_time)
print(min_time)
# 删除最大值和最小值
filtered_data = data[(data['Time'] != max_time) & (data['Time'] != min_time)]

# 计算去掉最大值和最小值后的平均时间
trimmed_average_time = filtered_data['Time'].mean()
print(f"去掉最大值和最小值后的平均时间: {trimmed_average_time}")

# 统计不同时间的合约地址个数
# time_counts = data['Time'].value_counts()

# 统计不同时间范围内的合约地址个数
# time_ranges = [(0, 1), (1, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30)]
time_ranges = [(i, i+1) for i in range(30)]
time_counts = []
for start, end in time_ranges:
    count = data[(data['Time'] >= start) & (data['Time'] < end)]['contract_creation_tx'].nunique()
    time_counts.append(count)
# 绘制柱状图
plt.figure(figsize=(5, 4))
# time_counts.plot(kind='bar')
# plt.bar(time_ranges, time_counts)
plt.bar([start for start, _ in time_ranges], time_counts)
plt.xlabel('Time(s)')
plt.ylabel('Number of Contracts')
plt.title('Processing Time Distribution')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('data/output/379_time_distribution.png')
# plt.show()