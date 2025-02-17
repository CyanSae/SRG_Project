# import pandas as pd
# import matplotlib.pyplot as plt

# # 读取 CSV 文件
# file_path = "data/output/379_processing_times2 copy.csv"  # 替换为你的文件名
# df = pd.read_csv(file_path)

# # 定义时间范围分类
# bins = [0, 0.1, 0.5, 1, float('inf')]
# labels = ["<0.1s", "0.1-0.5s", "0.5-1s", ">1s"]
# df['Time_Interval'] = pd.cut(df['Time'], bins=bins, labels=labels, right=False)

# # 统计各时间范围的分布
# time_distribution = df['Time_Interval'].value_counts(normalize=True) * 100

# # 打印结果
# print(time_distribution)

# # 绘制环形图
# plt.figure(figsize=(7, 7))
# colors = ['#5a92af', '#86c1d4', '#9cd9de', '#d9f9f4']
# wedges, texts, autotexts = plt.pie(
#     time_distribution,
#     # labels=time_distribution.index,
#     autopct='%1.1f%%',
#     pctdistance=0.65,
#     startangle=90,
#     colors=colors,
#     wedgeprops={'width': 0.4},
#     textprops={'fontsize': 12}
# )
# plt.legend(
#     wedges,
#     time_distribution.index,
#     title="Time Interval",
#     # loc="center left",
#     # bbox_to_anchor=(1, 0, 0.5, 1),  # 设置图例在饼图右侧
#     fontsize=12
# )
# # plt.pie(time_distribution, labels=time_distribution.index, autopct='%1.1f%%', startangle=90, colors=colors, wedgeprops={'width': 0.4})
# plt.title('Malicious Contracts', fontsize=20)
# # plt.show()
# plt.tight_layout()
# plt.savefig('data/output/379_time_distribution_chart.pdf', format='pdf', dpi=300)
# # plt.savefig('data/output/379_time_distribution_chart.png')

import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV 文件
file_path = "data/output/processing_times.csv"  # 替换为你的文件名
df = pd.read_csv(file_path)

# 定义时间范围分类和颜色映射
bins = [0, 0.1, 0.5, 1, float('inf')]
labels = ["<0.1s", "0.1-0.5s", "0.5-1s", ">1s"]
colors_mapping = {
    "<0.1s": "#5a92af",
    "0.1-0.5s": "#86c1d4",
    "0.5-1s": "#9cd9de",
    ">1s": "#d9f9f4"
}

# 对时间数据进行分类
df['Time_Interval'] = pd.cut(df['Time'], bins=bins, labels=labels, right=False)

# 统计各时间范围的分布，确保顺序与区间一致
time_distribution = df['Time_Interval'].value_counts(normalize=True) * 100
time_distribution = time_distribution.reindex(labels)  # 确保分布顺序与 labels 一致

# 根据分类生成颜色列表
colors = [colors_mapping[label] for label in labels]

# 绘制环形图
plt.figure(figsize=(7, 7))
wedges, texts, autotexts = plt.pie(
    time_distribution,
    autopct='%1.1f%%',
    pctdistance=0.65,
    startangle=90,
    colors=colors,
    wedgeprops={'width': 0.4},
    textprops={'fontsize': 12}
)

# 添加图例，绑定颜色与区间
plt.legend(
    wedges,
    labels,
    title="Time Interval",
    loc="upper right",
    fontsize=12
)

# 设置标题
plt.title('Benign Contracts', fontsize=20)

# 自动调整布局并保存为高分辨率 PDF
plt.tight_layout()
output_pdf = 'data/output/b_time_distribution_chart.pdf'
plt.savefig(output_pdf, format='pdf', dpi=300)
plt.show()

print(f"Chart saved as {output_pdf}")

