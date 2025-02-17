import os
import json
from collections import defaultdict
import csv

def count_node_names(directory):
    node_count = defaultdict(int)
    edge_count = defaultdict(int)
    file_count = 0
    file_stats = []  # 用于存储每个文件的统计结果

    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as f:
                graph_data = json.load(f)
                nodes = graph_data.get("nodes", {})
                edges = graph_data.get("edges", [])

                # 统计节点名称出现的次数
                for node_name in nodes.values():
                    if node_name.startswith("CONST"):
                        node_name = "CONST"
                    node_count[node_name] += 1

                # 统计边的数量
                for edge in edges:
                    edge_count[edge[2]] += 1
                file_count += 1

                file_stats.append({
                    "filename": filename,
                    "node_count": len(nodes),
                    "edge_count": len(edges)
                })

    # 将统计结果按降序排序
    sorted_node_count = dict(sorted(node_count.items(), key=lambda item: item[1], reverse=True))
    sorted_edge_count = dict(sorted(edge_count.items(), key=lambda item: item[0]))
    # 计算每一种操作码和边类型的平均值
    average_node_count = {node: count / file_count for node, count in sorted_node_count.items()}
    average_edge_count = {edge: count / file_count for edge, count in sorted_edge_count.items()}

    return sorted_node_count, sorted_edge_count, average_node_count, average_edge_count, file_stats


def save_to_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def save_to_csv(data, file_path):
    with open(file_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "node_count", "edge_count"])
        writer.writeheader()
        writer.writerows(data)

directory_path = '/home/sandra/projects/DATA/SOG_SET/benign'
output_file_path = 'data/output/benign_frequency.json'
output_csv_path = 'data/output/benign_file_stats.csv'

node_count, edge_count, average_node_count, average_edge_count, file_stats = count_node_names(directory_path)

# 将节点名称和边的数量合并为一个字典
result = {
    "node_frequency": node_count,
    "edge_frequency": edge_count,
    "average_node_frequency": average_node_count,
    "average_edge_frequency": average_edge_count
}

save_to_json(result, output_file_path)
print(f"Node count and edge count saved to {output_file_path}")

save_to_csv(file_stats, output_csv_path)
print(f"File statistics saved to {output_csv_path}")