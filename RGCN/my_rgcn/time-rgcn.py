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
EPOCH = 1000
H_DIM = 16
OUT_DIM = 2
BATCH_SIZE = 32
DROP_OUT = 0.7
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

with open('RGCN/processed_dataset/time/30_train_20231025_shuffled.pkl', 'rb') as f:
    train_dataset = pickle.load(f)
with open('RGCN/processed_dataset/time/30_test_20231028_shuffled.pkl', 'rb') as f:
    test_dataset = pickle.load(f)

opcodes = [
"CONST", "JUMPDEST", "ADD", "JUMP", "MSTORE", "JUMPI", "AND", "MLOAD", "ISZERO", "SUB", "REVERT", "SHL", "EQ", "SLOAD", "SHA3", "LT", "MUL", "RETURNDATASIZE", "CALLDATALOAD", "GT", "DIV", "CALLVALUE", "EXP", "SSTORE", "NOT", "CALLDATASIZE", "RETURN", "CALLER", "SLT", "RETURNDATACOPY", "OR", "LOG", "GAS", "EXTCODESIZE", "CODECOPY", "STOP", "CALL", "ADDRESS", "INVALID", "CALLDATACOPY", "STATICCALL", "SHR", "GASPRICE", "TIMESTAMP", "DELEGATECALL", "GASLIMIT", "NOP", "ADDMOD", "SIGNEXTEND", "BALANCE", "MOD", "SMOD", "SGT", "MSTORE8", "ORIGIN", "BYTE", "NUMBER", "MISSING", "SDIV", "CREATE2", "CALLCODE", "CREATE", "MULMOD", "EXTCODEHASH", "COINBASE", "SELFDESTRUCT", "CODESIZE", "XOR", "BLOCKHASH", "DIFFICULTY", "SAR", "EXTCODECOPY", "MSIZE", "PC"]
opcode_to_feature = {opcode: index for index, opcode in enumerate(opcodes)}
num_opcodes = len(opcode_to_feature)

print(f"Number of graphs in loaded dataset: {len(train_dataset)}")
# graph, label = dataset[0]
# print(f"Graph: {graph}")
# print(f"Label: {label}")
num_labels_1 = 0
for batched_graph, labels in train_dataset:
    num_labels_1 += (labels == 1).sum().item()

print(f"Number of labels 1: {num_labels_1}")
# dataset = dgl.data.GINDataset("PROTEINS", self_loop=True)

from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler

num_examples = len(train_dataset)
num_train = int(num_examples * 1)
# num_val = int(num_examples * 0.25)
# num_test = num_examples - num_train - num_val

train_sampler = SubsetRandomSampler(torch.arange(num_train))
# val_sampler = SubsetRandomSampler(torch.arange(num_train, num_train + num_val))
# test_sampler = SubsetRandomSampler(torch.arange(num_train + num_val, num_examples))
# test_sampler = SubsetRandomSampler(torch.arange(len(test_dataset)))

train_dataloader = GraphDataLoader(
    train_dataset, sampler=train_sampler, batch_size=BATCH_SIZE, drop_last=False
)
# val_dataloader = GraphDataLoader(
#     train_dataset, sampler=val_sampler, batch_size=BATCH_SIZE, drop_last=False
# )
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
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024  # 转换为 MB

train_losses = []
val_losses = []
train_acc = []
val_acc = []
best_val_loss = 100
best_epoch = 0
start=time.time()
initial_memory = get_memory_usage()
for epoch in range(EPOCH):
   
    model.train()
    num_correct = 0
    num_tests = 0
    epoch_loss = 0
    for batched_graph, labels in train_dataloader:
        # node_features = batched_graph.ndata['feat'].float()
        # edge_features = batched_graph.edata['feat'].float()
        batched_graph = batched_graph.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        pred = model(batched_graph, batched_graph.ndata['feat'], batched_graph.edata['etype'])
        num_correct += (pred.argmax(1) == labels).sum().item()
        num_tests += len(labels)
        loss = criterion(pred, labels)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_accuracy = num_correct / num_tests
    avg_loss = epoch_loss / len(train_dataloader)
    train_losses.append(avg_loss)
    train_acc.append(train_accuracy)

    # model.eval()
    # val_loss = 0
    # val_correct = 0
    # val_tests = 0
    # epoch_loss = 0
    # with torch.no_grad():
    #     for batched_graph, labels in val_dataloader:
    #         # node_features = batched_graph.ndata['feat'].float()
    #         # edge_features = batched_graph.edata['feat'].float()
    #         batched_graph = batched_graph.to(device)
    #         labels = labels.to(device)
    #         pred = model(batched_graph, batched_graph.ndata['feat'], batched_graph.edata['etype'])
    #         val_correct += (pred.argmax(1) == labels).sum().item()
    #         val_tests += len(labels)
    #         val_loss += criterion(pred, labels).item()

    # val_loss /= len(val_dataloader)
    # val_accuracy = val_correct / val_tests
    # val_losses.append(val_loss)
    # val_acc.append(val_accuracy)
    print(f"Epoch {epoch+1}: Train loss: {loss.item()}, Train accuracy: {train_accuracy}, num_correct: {num_correct}, num_tests: {num_tests}")
    # print(f"Epoch {epoch+1}: Validation loss: {val_loss}, Validation accuracy: {val_accuracy}")

    # if val_loss < best_val_loss:
    #     best_val_loss = val_loss
    #     best_model_state = model.state_dict()
    #     best_epoch = epoch
    #     # torch.save(model.state_dict(), f'model/trained_model/{epoch}_gcn_{time.time()}.pt')
end=time.time()
end_memory = get_memory_usage()
print(f"train time: {end-start}")
print(f"train memory: {end_memory} MB")

# torch.save(best_model_state, f'RGCN/model/trained_model/time_{len(train_dataset)}_{best_epoch}-{len(train_losses)}_gcn2_{time.time()}.pt')
# model.load_state_dict(best_model_state)


model.eval()
num_correct = 0
num_tests = 0
all_preds = []
all_labels = []
test_start=time.time()
test_start_memory = get_memory_usage()
wrong = []
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
test_end_memory = get_memory_usage()
print(f"test time: {test_end-test_start}")
print(f"test memory: {test_end_memory} MB")

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
print(wrong)


plt.figure(figsize=(10, 8))
df = pd.DataFrame(train_losses)
train_losses = list(np.hstack(df.rolling(5, min_periods=1).mean().values))
df1 = pd.DataFrame(val_losses)
# val_losses = list(np.hstack(df1.rolling(5, min_periods=1).mean().values))
plt.plot(range(len(train_losses)), train_losses, label='Train Loss')
# plt.plot(range(len(val_losses)), val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
# plt.show()
plt.savefig(f"RGCN/losses/with_val/time_{len(train_dataset)}_training_val_{len(train_losses)}_{time.time()}.png")


plt.figure(figsize=(10, 8))
df = pd.DataFrame(train_acc)
train_acc = list(np.hstack(df.rolling(5, min_periods=1).mean().values))
df1 = pd.DataFrame(val_acc)
# val_acc = list(np.hstack(df1.rolling(5, min_periods=1).mean().values))
plt.plot(range(len(train_acc)), train_acc, label='Train Accuracy')
# plt.plot(range(len(val_acc)), val_acc, label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy Curve')
plt.legend()
plt.savefig(f"RGCN/accs/with_val/time_{len(train_dataset)}_training_acc_{len(train_losses)}_{time.time()}.png")
