from datasets import load_dataset
import pandas as pd

# 加载数据集
dataset = load_dataset("forta/malicious-smart-contract-dataset")

# 将训练集转换为 DataFrame
df = pd.DataFrame(dataset['train'])

# 获取前一千行
df_first_1000 = df.head(10000)

# 保存为 CSV 文件
df_first_1000.to_csv('data/first_10000_rows.csv', index=False)