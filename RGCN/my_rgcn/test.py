import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import RelGraphConv
from torchmetrics.functional import accuracy
from dgl.nn import RelGraphConv
from torch.utils.data import DataLoader
from process_all import ContractGraphDataset
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

LR = 0.0001
EPOCH = 500
H_DIM = 16
OUT_DIM = 2
BATCH_SIZE = 8
DROP_OUT = 0.5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

with open('RGCN/dataset/367/compile_evasion.pkl', 'rb') as f:
    test_dataset = pickle.load(f)

opcodes = [
"CONST", "JUMPDEST", "ADD", "JUMP", "MSTORE", "JUMPI", "AND", "MLOAD", "ISZERO", "SUB", "REVERT", "SHL", "EQ", "SLOAD", "SHA3", "LT", "MUL", "RETURNDATASIZE", "CALLDATALOAD", "GT", "DIV", "CALLVALUE", "EXP", "SSTORE", "NOT", "CALLDATASIZE", "RETURN", "CALLER", "SLT", "RETURNDATACOPY", "OR", "LOG", "GAS", "EXTCODESIZE", "CODECOPY", "STOP", "CALL", "ADDRESS", "INVALID", "CALLDATACOPY", "STATICCALL", "SHR", "GASPRICE", "TIMESTAMP", "DELEGATECALL", "GASLIMIT", "NOP", "ADDMOD", "SIGNEXTEND", "BALANCE", "MOD", "SMOD", "SGT", "MSTORE8", "ORIGIN", "BYTE", "NUMBER", "MISSING", "SDIV", "CREATE2", "CALLCODE", "CREATE", "MULMOD", "EXTCODEHASH", "COINBASE", "SELFDESTRUCT", "CODESIZE", "XOR", "BLOCKHASH", "DIFFICULTY", "SAR", "EXTCODECOPY", "MSIZE", "PC"]
opcode_to_feature = {opcode: index for index, opcode in enumerate(opcodes)}
num_opcodes = len(opcode_to_feature)

print(f"Number of graphs in loaded dataset: {len(test_dataset)}")
# graph, label = dataset[0]
# print(f"Graph: {graph}")
# print(f"Label: {label}")
num_labels_1 = 0
for batched_graph, labels in test_dataset:
    num_labels_1 += (labels == 1).sum().item()

print(f"Number of labels 1: {num_labels_1}")
# dataset = dgl.data.GINDataset("PROTEINS", self_loop=True)

def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024  

from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler

test_dataloader = GraphDataLoader(
    test_dataset, batch_size=1, drop_last=False
)

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


model = RGCN(in_dim=num_opcodes, h_dim=H_DIM, out_dim=2, num_rels=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

best_model_state = torch.load("RGCN/model/trained_model/1367_rgcn2_1555_613-800_gcn2_1737288333.4359772.pt")
model.load_state_dict(best_model_state)

model.eval()
num_correct = 0
num_tests = 0
all_preds = []
all_labels = []
test_start=time.time()
wrong = []
test_start_memory = get_memory_usage()
with torch.no_grad():
    for batched_graph, labels in test_dataloader:
        # node_features = batched_graph.ndata['feat'].float()
        # edge_features = batched_graph.edata['feat'].float()
        batched_graph = batched_graph.to(device)
        labels = labels.to(device)
        pred = model(batched_graph, batched_graph.ndata['feat'], batched_graph.edata['etype'])
        num_correct += (pred.argmax(1) == labels).sum().item()
        if pred.argmax(1) != labels:
            wrong.append([pred.argmax(1), labels, batched_graph])
        num_tests += len(labels)
        all_preds.extend(pred.argmax(1).tolist())
        all_labels.extend(labels.tolist())
test_end=time.time()
print(f"test time: {test_end-test_start}")
print(f"test memory: {test_start_memory}")
test_accuracy = num_correct / num_tests
precision = precision_score(all_labels, all_preds, average='binary', zero_division = 0)
recall = recall_score(all_labels, all_preds, average='binary')
f1 = f1_score(all_labels, all_preds, average='binary')


print(f"Test accuracy: {test_accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {f1}")
print(f"num_correct: {num_correct}, num_tests: {num_tests}")

# TN FP
# FN TP
test_cm = confusion_matrix(all_labels, all_preds)
print("Test confusion matrix:")
print(test_cm)
for wrongs in wrong:
    print(wrong)

# plt.figure(figsize=(10, 8))
# df = pd.DataFrame(train_losses)
# train_losses = list(np.hstack(df.rolling(5, min_periods=1).mean().values))
# df1 = pd.DataFrame(val_losses)
# val_losses = list(np.hstack(df1.rolling(5, min_periods=1).mean().values))
# plt.plot(range(len(train_losses)), train_losses, label='Train Loss')
# plt.plot(range(len(val_losses)), val_losses, label='Val Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training Loss Curve')
# plt.legend()
# # plt.show()
# plt.savefig(f"RGCN/losses/with_val/1367_rgcn2_{len(train_dataset)}_training_val_6:2_{len(train_losses)}_{time.time()}.png")

# plt.figure(figsize=(10, 8))
# df = pd.DataFrame(train_acc)
# train_acc = list(np.hstack(df.rolling(5, min_periods=1).mean().values))
# df1 = pd.DataFrame(val_acc)
# val_acc = list(np.hstack(df1.rolling(5, min_periods=1).mean().values))
# plt.plot(range(len(train_acc)), train_acc, label='Train Accuracy')
# plt.plot(range(len(val_acc)), val_acc, label='Val Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.title('Training Accuracy Curve')
# plt.legend()
# plt.savefig(f"RGCN/accs/with_val/1367_rgcn2_{len(train_dataset)}_training_acc_6:2_{len(train_losses)}_{time.time()}.png")
