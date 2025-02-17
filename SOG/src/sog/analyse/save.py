# import pandas as pd

# # 输入和输出文件路径
# input_file = 'data/compile.csv'  # 替换为您的输入文件路径
# output_file = 'data/compile_5.csv'  # 替换为您的输出文件路径

# # 读取CSV文件
# try:
#     df = pd.read_csv(input_file)
    
#     # 检查是否存在 'contract_creation_tx' 列
#     if 'contract_name' in df.columns:
#         # 提取 'contract_creation_tx' 列并仅保留前 1200 行
#         df_contract_creation_tx = df[['contract_name']].head(1200)
        
#         # 保存到新的CSV文件
#         df_contract_creation_tx.to_csv(output_file, index=False)
#         print(f"'contract_name' 列的前 1200 行已成功保存到 {output_file}")
#     else:
#         print("输入文件中没有 'contract_name' 列")
# except Exception as e:
#     print(f"发生错误: {e}")

import json
import pandas as pd

# 输入和输出文件路径
input_file = 'data/advSC_evasion_bytecode.json'  # 替换为您的输入 JSON 文件路径
output_file = 'data/compile_evation.csv'  # 替换为您的输出 CSV 文件路径

try:
    # 读取 JSON 文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取 `creation_bytecode`
    records = []
    for key, value in data.items():
        if 'creation_bytecode' in value:
            records.append({'contract_name': key, 'creation_bytecode': value['creation_bytecode'], 'deployed_bytecode': value['deployed_bytecode']})
    
    # 转换为 DataFrame
    df = pd.DataFrame(records)
    
    # 保存到 CSV 文件
    df.to_csv(output_file, index=False)
    print(f"'creation_bytecode' 数据已成功保存到 {output_file}")
    
except Exception as e:
    print(f"发生错误: {e}")

