# # import os
# # import csv

# # folder_path = 'RGCN/dataset/verified_contract_bins'  

# # file_names = os.listdir(folder_path)

# # # 创建并写入 CSV 文件
# # with open('RGCN/dataset/ponzi/non-ponzi.csv', mode='w', newline='') as file:
# #     writer = csv.DictWriter(file, fieldnames=['contract_address', 'malicious'])
# #     writer.writeheader()
    
# #     # 遍历文件名，写入到 CSV 文件
# #     for file_name in file_names:
# #         # 如果是文件且不是目录
# #         if os.path.isfile(os.path.join(folder_path, file_name)):
# #             writer.writerow({'contract_address': file_name, 'malicious': 0})

# # print("文件名已保存")

# import os
# import csv

# # 定义目标文件夹路径
# folder_path = 'RGCN/dataset/verified_contract_bins'  # 请替换为你的文件夹路径

# # 获取文件夹下的所有文件
# file_names = os.listdir(folder_path)

# # 创建并写入 CSV 文件
# with open('RGCN/dataset/ponzi/non-ponzi.csv', mode='w', newline='') as file:
#     writer = csv.DictWriter(file, fieldnames=['contract_address', 'malicious', 'bytecode'])
#     writer.writeheader()

#     # 遍历文件名，读取每个文件的二进制内容并写入到 CSV 文件
#     for file_name in file_names:
#         file_path = os.path.join(folder_path, file_name)
        
#         # 如果是文件且不是目录
#         if os.path.isfile(file_path):
#             # 去掉 .bin 后缀（如果存在）
#             contract_address = file_name
#             if contract_address.endswith('.bin'):
#                 contract_address = contract_address[:-4]  # 去掉 .bin 后缀

#             with open(file_path, 'rb') as f:
#                 bytecode = f.read()
#                 # bytecode = bytecode[2:-1]  # 去掉开头的 b' 和结尾的 '

#             # 将文件名、恶意标志和二进制内容写入到 CSV 文件
#             writer.writerow({'contract_address': contract_address, 'malicious': 0, 'bytecode': bytecode})

# print("文件名、恶意标志和二进制内容已保存")


import csv
import sys

def find_creation_bytecode(csv_file_path, target_contract_address, output_file_path):
    # 增加字段大小限制
    max_int = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int = int(max_int / 10)

    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        csvreader = csv.DictReader(csvfile)
        
        for row in csvreader:
            if row['contract_address'] == target_contract_address:
                creation_bytecode = row['creation_bytecode']
                
                with open(output_file_path, mode='w', encoding='utf-8') as output_file:
                    output_file.write(creation_bytecode)
                
                print(f"Creation bytecode for contract address {target_contract_address} has been saved to {output_file_path}.")
                return
        
        print(f"Contract address {target_contract_address} not found in the CSV file.")

# 使用示例
csv_file_path = 'SOG/data/large/first_2000_rows.csv'  # 替换为你的CSV文件路径
target_contract_address = '0x020810D775fC019480CD56ECb960389d3477039D'  # 替换为你要查找的合约地址
output_file_path = 'SOG/examples/322.hex'

find_creation_bytecode(csv_file_path, target_contract_address, output_file_path)