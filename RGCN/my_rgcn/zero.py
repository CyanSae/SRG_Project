import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import RelGraphConv
from torchmetrics.functional import accuracy
from dgl.nn import RelGraphConv
from torch.utils.data import DataLoader
import pickle
import dgl.data
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import time
import pandas as pd
import numpy as np
import psutil
import os
import json

LR = 0.0001
EPOCH = 500
H_DIM = 16
OUT_DIM = 2
BATCH_SIZE = 8
DROP_OUT = 0.5

opcodes = [
"CONST", "JUMPDEST", "ADD", "JUMP", "MSTORE", "JUMPI", "AND", "MLOAD", "ISZERO", "SUB", "REVERT", "SHL", "EQ", "SLOAD", "SHA3", "LT", "MUL", "RETURNDATASIZE", "CALLDATALOAD", "GT", "DIV", "CALLVALUE", "EXP", "SSTORE", "NOT", "CALLDATASIZE", "RETURN", "CALLER", "SLT", "RETURNDATACOPY", "OR", "LOG", "GAS", "EXTCODESIZE", "CODECOPY", "STOP", "CALL", "ADDRESS", "INVALID", "CALLDATACOPY", "STATICCALL", "SHR", "GASPRICE", "TIMESTAMP", "DELEGATECALL", "GASLIMIT", "NOP", "ADDMOD", "SIGNEXTEND", "BALANCE", "MOD", "SMOD", "SGT", "MSTORE8", "ORIGIN", "BYTE", "NUMBER", "MISSING", "SDIV", "CREATE2", "CALLCODE", "CREATE", "MULMOD", "EXTCODEHASH", "COINBASE", "SELFDESTRUCT", "CODESIZE", "XOR", "BLOCKHASH", "DIFFICULTY", "SAR", "EXTCODECOPY", "MSIZE", "PC"]
opcode_to_feature = {opcode: index for index, opcode in enumerate(opcodes)}
num_opcodes = len(opcode_to_feature)

class RGCN(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, num_rels):
        super().__init__()
        # Two-layer RGCN
        self.conv1 = RelGraphConv(
            in_dim,
            h_dim,
            num_rels,
            regularizer="basis",
            num_bases=num_rels,
            self_loop=False,
        )
        self.conv2 = RelGraphConv(
            h_dim,
            out_dim,
            num_rels,
            regularizer="basis",
            num_bases=num_rels,
            self_loop=False,
        )
        self.dropout = DROP_OUT
        self.fc = nn.Linear(out_dim, 2)  

    def forward(self, g, features, etypes):
        h = F.relu(self.conv1(g, features, etypes))
        h = self.conv2(g, h, etypes)
        h = F.dropout(h, self.dropout, training=self.training)
        
        with g.local_scope():
            g.ndata['h'] = h
            hg = dgl.mean_nodes(g, 'h')  
        return self.fc(hg)


model = RGCN(in_dim=num_opcodes, h_dim=H_DIM, out_dim=2, num_rels=3)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

best_model_state = torch.load("RGCN/model/trained_model/zeroday_rgcn_1555_989-1000-1743433653.0164285.pt")
model.load_state_dict(best_model_state)
model.eval()

json_dir = '/home/sandra/DATA/SOG_SET/zeroday_16'

predicted_as_1 = []

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

    return g

for filename in os.listdir(json_dir):
    if filename.endswith('.json'):
        file_path = os.path.join(json_dir, filename)
        
        g = process_json_file(file_path) 
        
        if g is not None:
            with torch.no_grad():
                pred = model(g, g.ndata['feat'], g.edata['etype'])
                predicted_label = pred.argmax(1).item()
                
                if predicted_label == 1:
                    print(f'File {filename} is predicted as 1.')
                    predicted_as_1.append(filename)

output_file = 'RGCN/output/predicted_as_1_files.txt'
with open(output_file, 'w') as f:
    for filename in predicted_as_1:
        f.write(f"{filename}\n")

print(f"Predicted as 1 filenames are saved to {output_file}")
