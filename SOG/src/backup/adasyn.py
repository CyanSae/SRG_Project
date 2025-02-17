import numpy as np
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import ADASYN

# 示例：语义图特征提取
def extract_features_from_graph(graph):
    # 将节点和边的特征提取为一个整体向量
    node_features = [graph.nodes[n]['feature'] for n in graph.nodes]
    edge_features = [(u, v, graph.edges[u, v]['feature']) for u, v in graph.edges]
    
    # Flatten the features
    flat_node_features = np.array(node_features).flatten()
    flat_edge_features = np.array([f[2] for f in edge_features])
    
    # Combine node and edge features
    combined_features = np.concatenate([flat_node_features, flat_edge_features])
    return combined_features

# 示例：特征向量还原为语义图
def restore_graph_from_features(feature_vector, num_nodes, edge_list):
    # 将特征向量分解为节点和边特征
    node_features = feature_vector[:num_nodes]
    edge_features = feature_vector[num_nodes:]
    
    # 构建新图
    new_graph = nx.DiGraph()
    for i, feature in enumerate(node_features):
        new_graph.add_node(i, feature=int(feature))
    
    for i, (u, v) in enumerate(edge_list):
        new_graph.add_edge(u, v, feature=int(edge_features[i]))
    
    return new_graph

# 示例：合成新语义图
def synthesize_semantic_graphs(graphs, labels):
    # 提取特征向量
    features = np.array([extract_features_from_graph(g) for g in graphs])
    labels = np.array(labels)
    
    # 特征归一化
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    
    # ADASYN合成
    ada = ADASYN(sampling_strategy='minority', random_state=42)
    features_resampled, labels_resampled = ada.fit_resample(features_scaled, labels)
    
    # 将合成的特征还原为语义图
    num_nodes = len(graphs[0].nodes)
    edge_list = list(graphs[0].edges)
    synthesized_graphs = [
        restore_graph_from_features(scaler.inverse_transform([f])[0], num_nodes, edge_list)
        for f in features_resampled[len(graphs):]
    ]
    
    return synthesized_graphs

# 示例：输入语义图和标签
# 创建示例语义图
G1 = nx.DiGraph()
G1.add_node(0, feature=10)
G1.add_node(1, feature=20)
G1.add_edge(0, 1, feature=1)

G2 = nx.DiGraph()
G2.add_node(0, feature=30)
G2.add_node(1, feature=40)
G2.add_edge(0, 1, feature=2)

G3 = nx.DiGraph()
G3.add_node(0, feature=50)
G3.add_node(1, feature=60)
G3.add_edge(0, 1, feature=0)
graphs = [G1, G2, G3]
labels = [0, 1, 0] 

# 合成新语义图
synthesized_graphs = synthesize_semantic_graphs(graphs, labels)

# 打印结果
for i, g in enumerate(synthesized_graphs, 1):
    print(f"合成语义图 {i}:")
    print("节点特征:", [g.nodes[n]['feature'] for n in g.nodes])
    print("边特征:", [(u, v, g.edges[u, v]['feature']) for u, v in g.edges])
