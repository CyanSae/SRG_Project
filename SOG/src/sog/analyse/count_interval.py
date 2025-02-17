import pandas as pd

# 读取 CSV 文件
file_path = "data/output/benign_file_stats.csv"  # 替换为你的文件路径
df = pd.read_csv(file_path)

# 定义函数统计非零最小值、最大值及区间分布和百分比
def analyze_column(data, column, bins, labels):
    # 筛选 column > 0 的行
    non_zero_data = data[data[column] > 0]

    # 计算最小值和最大值
    min_value = non_zero_data[column].min()
    max_value = non_zero_data[column].max()

    # 计算区间分布及其百分比
    non_zero_data['range'] = pd.cut(non_zero_data[column], bins=bins, labels=labels, right=False)
    range_distribution = non_zero_data['range'].value_counts().sort_index()
    range_percentage = (range_distribution / len(non_zero_data) * 100).round(2)

    # 将分布和百分比合并
    distribution_with_percentage = pd.DataFrame({
        "count": range_distribution,
        "percentage": range_percentage
    })

    return min_value, max_value, distribution_with_percentage

# 定义区间范围
bins = [1, 100, 1000, 5000, 10000, float('inf')]
labels = ['1-100', '101-1000', '1001-5000', '5001-10000', '>10000']

# 分别统计 node_count 和 edge_count
node_min, node_max, node_distribution = analyze_column(df, 'node_count', bins, labels)
edge_min, edge_max, edge_distribution = analyze_column(df, 'edge_count', bins, labels)

# 输出结果
print(f"Non-zero minimum node_count: {node_min}")
print(f"Non-zero maximum node_count: {node_max}")
print("Node count range distribution:")
print(node_distribution)

print(f"Non-zero minimum edge_count: {edge_min}")
print(f"Non-zero maximum edge_count: {edge_max}")
print("Edge count range distribution:")
print(edge_distribution)

# 保存区间分布到 CSV 文件
# output_file_path_nodes = "data/output/node_count_distribution.csv"
# output_file_path_edges = "data/output/edge_count_distribution.csv"

# node_distribution.to_csv(output_file_path_nodes)
# edge_distribution.to_csv(output_file_path_edges)

# print(f"Node count distribution saved to {output_file_path_nodes}")
# print(f"Edge count distribution saved to {output_file_path_edges}")
