from pyvis.network import Network
import json
import math
directory = '/home/sandra/DATA/SOG_SET/360connection/'
file = '0xf4f51d7c1536d6b7729803ebe87aad5baad0053a505ecba88632a453f38cb6cc'
file_path = directory + file + '.json'

NODE_COUNT = 5000  # 实际节点数量
BASE_NODE_DISTANCE = 150  # 基础间距
repulsion_distance = BASE_NODE_DISTANCE * math.sqrt(NODE_COUNT/50)
stabilization_iterations = min(1000, NODE_COUNT * 10)

def process_json_file(file_path):
    with open(file_path, 'r') as f:
        graph_data = json.load(f)

    nodes = graph_data["nodes"]
    edges = graph_data["edges"]

    if not nodes or not edges:
        print(f"Empty graph found in file: {file_path}")
        return None
    return graph_data
# 你的 JSON 数据
# graph_data = {
#     "nodes": {
#         "1": "MISSING",
#         "0": "JUMPI",
#         "55": "MULMOD",
#         "53": "MLOAD",
#         "51": "MSTORE8",
#         "56": "RETURN"
#     },
#     "edges": [
#         [1, 0, 1],
#         [55, 53, 2],
#         [53, 51, 2],
#         [56, 55, 0],
#         [51, 1, 3]
#     ],
#     "node_count": 6,
#     "edge_count": 5
# }

edge_colors = {
    0: {"color": "#8CBA54", "label": "Data"},  # 红色
    1: {"color": "#4091CF", "label": "Control"},  # 绿色
    2: {"color": "#E1703C", "label": "Effect"},  # 蓝色
    3: {"color": "#818181", "label": "Connection"},  # 紫色
    "default": {"color": "#818181", "label": "Other"}  # 灰色
}
graph_data = process_json_file(file_path)
# 创建 pyvis 网络图
net = Network(height="100vh", width="100%", notebook=False, directed=True)

# 添加节点
for node_id, label in graph_data["nodes"].items():
    net.add_node(node_id, label=label, color="#97c2fc", font="12", size="20")

# 添加边（处理可能的字符串/整数类型）
for edge in graph_data["edges"]:
    source = str(edge[0])
    target = str(edge[1])
    edge_type = edge[2]
    
    # 获取边配置
    config = edge_colors.get(edge_type, edge_colors["default"])
    
    net.add_edge(
        source, 
        target,
        title=f"Edge Type: {edge_type}",
        label=config["label"],  # 显示在边上的文字
        color=config["color"],    # 边颜色
        width=2,                 # 边粗细
        arrows="to",             # 箭头方向
        smooth=False             # 直线连接
    )

# net.set_options("""
# {
#   "physics": {
#     "enabled": true,
#     "stabilization": {
#       "iterations": {stabilization_iterations},
#     },
#     "repulsion": {
#       "nodeDistance": 150
#     }
#   },
#   "edges": {
#     "font": {
#       "size": 10,
#       "align": "middle"
#     },
#     "arrowStrikethrough": false
#   }
# }
# """)

physics_config = f"""
{{
  "physics": {{
    "enabled": true,
    "stabilization": {{
      "enabled": true,
      "iterations": {stabilization_iterations},
      "updateInterval": 25,
      "fit": true
    }},
    "solver": "barnesHut",
    "barnesHut": {{
      "gravitationalConstant": -1500,
      "centralGravity": 0.3,
      "springLength": {repulsion_distance},
      "springConstant": 0.01,
      "damping": 0.3,
      "avoidOverlap": 0.2
    }},
    "maxVelocity": 10,
    "minVelocity": 0.5,
    "timestep": 0.4
  }},
  "edges": {{
    "font": {{
      "size": 10,
      "align": "middle"
    }},
    "arrowStrikethrough": false
  }},
  "nodes": {{
    "scaling": {{
      "min": 20,
      "max": 40
    }}
  }}
}}
"""
net.set_options(physics_config)

# 生成 HTML 文件
net.save_graph(f"SOG/vis/{file}.html")