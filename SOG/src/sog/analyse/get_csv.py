import pandas as pd

# 加载原始 CSV 文件
df = pd.read_csv('SOG/bytecode/phish2.csv')  # 替换成你的文件名，如 'contracts_with_bytecode.csv'

# 选择需要的两列
subset_df = df[['contract_address', 'malicious']]

# 保存为新的 CSV 文件
subset_df.to_csv('SOG/dataset/phish_contract.csv', index=False)

print("✅ 已成功提取并保存")
