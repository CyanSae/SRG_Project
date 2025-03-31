import os
import json
import pickle
import dgl
import torch
import pandas as pd
from dgl.data import DGLDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

# HOME_PATH = '/home/sandra/projects/'

json_dir = '/home/sandra/DATA/SOG_SET/2367'
csv_file_path = 'RGCN/shuffled_dataset/creation_1346_shuffled.csv'

labels_df = pd.read_csv(csv_file_path)
labels_dict = labels_df.set_index('contract_creation_tx')['malicious'].to_dict()

def process_json_file(file_path):
    with open(file_path, 'r') as f:
        graph_data = json.load(f)

    nodes = graph_data["nodes"]
    edges = graph_data["edges"]

    if not nodes or not edges:
        print(f"Empty graph found in file: {file_path}")
        return None, None

    node_map = {int(node_id): idx for idx, node_id in enumerate(nodes.keys())}

    src_nodes = [node_map[edge[0]] for edge in edges]
    dst_nodes = [node_map[edge[1]] for edge in edges]
    g = dgl.graph((src_nodes, dst_nodes))

    opcodes = [
    "CONST", "JUMPDEST", "ADD", "JUMP", "MSTORE", "JUMPI", "AND", "MLOAD", "ISZERO", "SUB", "REVERT", "SHL", "EQ", "SLOAD", "SHA3", "LT", "MUL", "RETURNDATASIZE", "CALLDATALOAD", "GT", "DIV", "CALLVALUE", "EXP", "SSTORE", "NOT", "CALLDATASIZE", "RETURN", "CALLER", "SLT", "RETURNDATACOPY", "OR", "LOG", "GAS", "EXTCODESIZE", "CODECOPY", "STOP", "CALL", "ADDRESS", "INVALID", "CALLDATACOPY", "STATICCALL", "SHR", "GASPRICE", "TIMESTAMP", "DELEGATECALL", "GASLIMIT", "NOP", "ADDMOD", "SIGNEXTEND", "BALANCE", "MOD", "SMOD", "SGT", "MSTORE8", "ORIGIN", "BYTE", "NUMBER", "MISSING", "SDIV", "CREATE2", "CALLCODE", "CREATE", "MULMOD", "EXTCODEHASH", "COINBASE", "SELFDESTRUCT", "CODESIZE", "XOR", "BLOCKHASH", "DIFFICULTY", "SAR", "EXTCODECOPY", "MSIZE", "PC"]

    opcode_to_feature = {opcode: index for index, opcode in enumerate(opcodes)}

    # num_nodes = g.num_nodes()

    num_opcodes = len(opcode_to_feature)

    # one-hot
    node_features = []
    for node_id, opcode in nodes.items():
        feature_idx = opcode_to_feature.get(opcode, 0) 
        node_features.append(feature_idx)
    node_features = F.one_hot(torch.tensor(node_features), num_classes=num_opcodes).float()
    g.ndata['feat'] = node_features

    edge_types = torch.tensor([edge[2] for edge in edges]).long()
    g.edata['etype'] = edge_types

    
    contract_address = os.path.splitext(os.path.basename(file_path))[0]

    label = 1 if labels_dict.get(contract_address, None) == 1 else 0
    # label = labels_dict.get(contract_address, 0)
    return g, label

class ContractGraphDataset(DGLDataset):
    def __init__(self, json_dir):
        self.json_dir = json_dir
        super().__init__(name='contract_graph')

    def process(self):
        self.graphs = []
        self.labels = []
        self.metadata = []
        for index, row in labels_df.iterrows():
            file_name = row['contract_creation_tx']
        # for file_name in os.listdir(self.json_dir):
            # if file_name.endswith('.json'):
            file_path = os.path.join(self.json_dir, file_name+'.json')
            if not os.path.exists(file_path):
                # print(f"文件不存在: {file_path}")
                continue
            g, label = process_json_file(file_path)
            if g is not None:
                self.graphs.append(g)
                self.labels.append(label)
                self.metadata.append(row['contract_creation_tx'])
        self.labels = torch.tensor(self.labels)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx], self.metadata[idx]

    def __len__(self):
        return len(self.graphs)

dataset = ContractGraphDataset(json_dir)

def collate_fn(batch):
    graphs, labels = map(list, zip(*batch))
    batched_graph = dgl.batch(graphs)
    labels = torch.tensor(labels)
    return batched_graph, labels

# dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn, shuffle = True)

# for batched_graph, labels in dataloader:
#     print("Batched graph:", batched_graph)
#     print("Labels:", labels)
print(f"Number of graphs in loaded dataset: {len(dataset)}")

# graph, label, metadata = dataset[899]
# print(f"Graph: {graph}")
# print(f"Label: {label}")
# print(f"Metadata: {metadata}")

num_labels_1 = 0
for batched_graph, labels, metadata in dataset:
    num_labels_1 += (labels == 1).sum().item()
print(f"Number of labels 1: {num_labels_1}")

with open('RGCN/processed_dataset/creation_1346_shuffled.pkl', 'wb') as f:
    pickle.dump(dataset, f)