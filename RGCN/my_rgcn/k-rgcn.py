from sklearn.model_selection import KFold
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import RelGraphConv
from dgl.dataloading import GraphDataLoader
from ContractGraphDataset import ContractGraphDataset
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import time
import pandas as pd
import numpy as np
import pickle
from torch.utils.data.sampler import SubsetRandomSampler

NUM_FOLDS = 5
BATCH_SIZE = 16
EPOCH = 1200
LR = 0.0001
H_DIM = 8
DROP_OUT = 0.5

# Initialize KFold
kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

fold_metrics = {
    "accuracy": [],
    "precision": [],
    "recall": [],
    "f1": []
}

opcodes = [
"CONST", "JUMPDEST", "ADD", "JUMP", "MSTORE", "JUMPI", "AND", "MLOAD", "ISZERO", "SUB", "REVERT", "SHL", "EQ", "SLOAD", "SHA3", "LT", "MUL", "RETURNDATASIZE", "CALLDATALOAD", "GT", "DIV", "CALLVALUE", "EXP", "SSTORE", "NOT", "CALLDATASIZE", "RETURN", "CALLER", "SLT", "RETURNDATACOPY", "OR", "LOG", "GAS", "EXTCODESIZE", "CODECOPY", "STOP", "CALL", "ADDRESS", "INVALID", "CALLDATACOPY", "STATICCALL", "SHR", "GASPRICE", "TIMESTAMP", "DELEGATECALL", "GASLIMIT", "NOP", "ADDMOD", "SIGNEXTEND", "BALANCE", "MOD", "SMOD", "SGT", "MSTORE8", "ORIGIN", "BYTE", "NUMBER", "MISSING", "SDIV", "CREATE2", "CALLCODE", "CREATE", "MULMOD", "EXTCODEHASH", "COINBASE", "SELFDESTRUCT", "CODESIZE", "XOR", "BLOCKHASH", "DIFFICULTY", "SAR", "EXTCODECOPY", "MSIZE", "PC"]

opcode_to_feature = {opcode: index for index, opcode in enumerate(opcodes)}

num_opcodes = len(opcode_to_feature)

with open('RGCN/dataset/367/creation_1367_shuffled.pkl', 'rb') as f:
    dataset = pickle.load(f)

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
        self.fc = nn.Linear(out_dim, 2)  # 二分类输出

    def forward(self, g, features, etypes):
        h = F.relu(self.conv1(g, features, etypes))
        h = self.conv2(g, h, etypes)
        h = F.dropout(h, self.dropout, training=self.training)
        
        # 整图表示：聚合节点特征
        with g.local_scope():
            g.ndata['h'] = h
            hg = dgl.mean_nodes(g, 'h')  # 整图表示
        return self.fc(hg)  # 整图分类
    
# Start cross-validation
for fold, (train_val_idx, test_idx) in enumerate(kf.split(dataset)):
    print(f"Starting fold {fold + 1}/{NUM_FOLDS}")

    # Split the dataset into training+validation and testing sets
    train_val_set = [dataset[i] for i in train_val_idx]
    test_set = [dataset[i] for i in test_idx]

    # Further split training+validation into training and validation sets
    num_train = int(len(train_val_set) * 0.8)
    train_set = train_val_set[:num_train]
    val_set = train_val_set[num_train:]

    # Create dataloaders
    train_dataloader = GraphDataLoader(train_set, batch_size=BATCH_SIZE, drop_last=False)
    val_dataloader = GraphDataLoader(val_set, batch_size=BATCH_SIZE, drop_last=False)
    test_dataloader = GraphDataLoader(test_set, batch_size=BATCH_SIZE, drop_last=False)

    # Reset the model for this fold
    model = RGCN(in_dim=num_opcodes, h_dim=H_DIM, out_dim=2, num_rels=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    best_val_loss = float('inf')
    best_model_state = None
    for epoch in range(EPOCH):
        model.train()
        for batched_graph, labels in train_dataloader:
            optimizer.zero_grad()
            pred = model(batched_graph, batched_graph.ndata['feat'], batched_graph.edata['etype'])
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()

        # Validate the model
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batched_graph, labels in val_dataloader:
                pred = model(batched_graph, batched_graph.ndata['feat'], batched_graph.edata['etype'])
                val_loss += criterion(pred, labels).item()
        val_loss /= len(val_dataloader)

        # Save the best model for this fold
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

    # Test the model
    model.load_state_dict(best_model_state)
    model.eval()
    all_preds, all_labels = [], []
    wrong = []
    with torch.no_grad():
        for batched_graph, labels in test_dataloader:
            pred = model(batched_graph, batched_graph.ndata['feat'], batched_graph.edata['etype'])
            # if pred.argmax(1) != labels:
            #     wrong.append([pred.argmax(1), labels, batched_graph])
            all_preds.extend(pred.argmax(1).tolist())
            all_labels.extend(labels.tolist())

    # Calculate metrics for this fold
    # accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')

    print(f"Fold {fold + 1} - Precision: {precision}, Recall: {recall}, F1: {f1}")
    # 测试集的混淆矩阵
    # TN FP
    # FN TP
    test_cm = confusion_matrix(all_labels, all_preds)
    print("Test confusion matrix:")
    print(test_cm)
    print(wrong)
    # fold_metrics["accuracy"].append(accuracy)
    fold_metrics["precision"].append(precision)
    fold_metrics["recall"].append(recall)
    fold_metrics["f1"].append(f1)

# Print overall metrics
print("Cross-validation results:")
for metric, values in fold_metrics.items():
    mean_value = np.mean(values)
    std_value = np.std(values)
    print(f"{metric.capitalize()}: Mean = {mean_value:.4f}, Std = {std_value:.4f}")
