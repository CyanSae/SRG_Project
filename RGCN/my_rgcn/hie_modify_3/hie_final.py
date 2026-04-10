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
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import os
import json
import seaborn as sns
import copy
LR = 0.0001
EPOCH = 400
H_DIM = 32
OUT_DIM = 9
BATCH_SIZE = 32
DROP_OUT = 0.3
# 层次化条件分类头
# 条件loss
# 试图记录了分类错误的creation tx
with open('RGCN/processed_dataset/hie/hie_others_shuffled1.pkl', 'rb') as f:
    dataset = pickle.load(f)
# Opcodes and related setup
opcodes = [
    "CONST", "JUMPDEST", "ADD", "JUMP", "MSTORE", "JUMPI", "AND", "MLOAD", "ISZERO", "SUB", "REVERT", "SHL",
    "EQ", "SLOAD", "SHA3", "LT", "MUL", "RETURNDATASIZE", "CALLDATALOAD", "GT", "DIV", "CALLVALUE", "EXP", "SSTORE",
    "NOT", "CALLDATASIZE", "RETURN", "CALLER", "SLT", "RETURNDATACOPY", "OR", "LOG", "GAS", "EXTCODESIZE", "CODECOPY",
    "STOP", "CALL", "ADDRESS", "INVALID", "CALLDATACOPY", "STATICCALL", "SHR", "GASPRICE", "TIMESTAMP", "DELEGATECALL",
    "GASLIMIT", "NOP", "ADDMOD", "SIGNEXTEND", "BALANCE", "MOD", "SMOD", "SGT", "MSTORE8", "ORIGIN", "BYTE", "NUMBER",
    "MISSING", "SDIV", "CREATE2", "CALLCODE", "CREATE", "MULMOD", "EXTCODEHASH", "COINBASE", "SELFDESTRUCT", "CODESIZE",
    "XOR", "BLOCKHASH", "DIFFICULTY", "SAR", "EXTCODECOPY", "MSIZE", "PC"
]
opcode_to_feature = {opcode: index for index, opcode in enumerate(opcodes)}
num_opcodes = len(opcode_to_feature)
print(f"Number of graphs in loaded dataset: {len(dataset)}")
# def create_data_loaders(dataset, batch_size=BATCH_SIZE, train_ratio=0.6, val_ratio=0.2):
#     num_examples = len(dataset)
#     num_train = int(num_examples * train_ratio)
#     num_val = int(num_examples * val_ratio)
#     train_sampler = SubsetRandomSampler(torch.arange(num_train))
#     val_sampler = SubsetRandomSampler(torch.arange(num_train, num_train + num_val))
#     test_sampler = SubsetRandomSampler(dataset, torch.arange(num_train + num_val, num_examples))
#     train_dataloader = GraphDataLoader(dataset, sampler=train_sampler, batch_size=batch_size, drop_last=False)
#     val_dataloader = GraphDataLoader(dataset, sampler=val_sampler, batch_size=batch_size, drop_last=False)
#     test_dataloader = GraphDataLoader(dataset, sampler=test_sampler, batch_size=1, drop_last=False)
    
#     return train_dataloader, val_dataloader, test_dataloader
def extract_l3_labels(dataset):
    l3_labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        if torch.is_tensor(label):
            l3_labels.append(int(label[2].item()))
        else:
            l3_labels.append(int(label[2]))
    return np.array(l3_labels)
def create_data_loaders(dataset, batch_size=BATCH_SIZE, train_ratio=0.6, val_ratio=0.2, random_state=42):
    num_examples = len(dataset)
    indices = np.arange(num_examples)
    # ===== 按 L3 分层划分 =====
    l3_labels = extract_l3_labels(dataset)
    train_idx, temp_idx = train_test_split(
        indices,
        train_size=train_ratio,
        stratify=l3_labels,
        random_state=random_state,
        shuffle=True
    )
    temp_ratio = 1.0 - train_ratio
    val_in_temp = val_ratio / temp_ratio
    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=val_in_temp,
        stratify=l3_labels[temp_idx],
        random_state=random_state,
        shuffle=True
    )
    train_sampler = SubsetRandomSampler(train_idx.tolist())
    val_sampler = SubsetRandomSampler(val_idx.tolist())
    # 测试集不要随机 sampler，保持顺序稳定，方便按索引回查 contract_creation_tx
    test_subset = Subset(dataset, test_idx.tolist())
    train_dataloader = GraphDataLoader(dataset, sampler=train_sampler, batch_size=batch_size, drop_last=False)
    val_dataloader = GraphDataLoader(dataset, sampler=val_sampler, batch_size=batch_size, drop_last=False)
    test_dataloader = GraphDataLoader(test_subset, batch_size=1, shuffle=False, drop_last=False)
    return train_dataloader, val_dataloader, test_dataloader, test_subset
# train_dataloader, val_dataloader, test_dataloader = create_data_loaders(dataset)
train_dataloader, val_dataloader, test_dataloader, test_subset = create_data_loaders(dataset)
class HierarchicalRGCN(nn.Module):
    def __init__(self, in_dim, h_dim, num_rels):
        super().__init__()
        self.conv1 = RelGraphConv(
            in_dim, h_dim, num_rels,
            regularizer="basis",
            num_bases=num_rels,
            self_loop=False
        )
        self.conv2 = RelGraphConv(
            h_dim, h_dim, num_rels,
            regularizer="basis",
            num_bases=num_rels,
            self_loop=False
        )
        self.dropout = nn.Dropout(DROP_OUT)
        # 层次化条件分类头
        self.fc_l1 = nn.Linear(h_dim, 2)   # benign / malicious
        self.fc_l2 = nn.Linear(h_dim, 2)   # attack / fraud（仅在 malicious 条件下有意义）
        self.fc_attack = nn.Linear(h_dim, 5)  # reentrancy / logic error / price manipulation / access control / other
        self.fc_fraud  = nn.Linear(h_dim, 3)  # ponzi / honeypot / phishing
    def forward(self, g, feat, etype):
        h = F.relu(self.conv1(g, feat, etype))
        h = self.conv2(g, h, etype)
        h = self.dropout(h)
        with g.local_scope():
            g.ndata['h'] = h
            hg = dgl.mean_nodes(g, 'h')
        out1 = self.fc_l1(hg)
        out2 = self.fc_l2(hg)      # P(attack / fraud | malicious)
        out_attack = self.fc_attack(hg)
        out_fraud = self.fc_fraud(hg)
        return out1, out2, out_attack, out_fraud
################################
# 初始化
################################
# class FocalLoss(nn.Module):
#     def __init__(self, weight=None, gamma=2.0):
#         super().__init__()
#         self.weight = weight
#         self.gamma = gamma
#     def forward(self, logits, targets):
#         ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
#         pt = torch.exp(-ce_loss)
#         focal_loss = ((1 - pt) ** self.gamma) * ce_loss
#         return focal_loss.mean()
def init_model(num_opcodes, num_rels=4):
    in_dim = num_opcodes
    model = HierarchicalRGCN(
        in_dim,
        H_DIM,
        num_rels
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )
    # ===== Label Smoothing =====
    label_smoothing = 0.05
    # ===== Class weights =====
    l1_weights = torch.tensor([1.0, 1100 / 1326])
    # 条件式 L2: attack / fraud（只在 malicious 上训练）
    l2_weights = torch.tensor([
        965 / 361,   # attack
        1.0          # fraud
    ])
    # fraud 子头: [ponzi, honeypot, phishing] -> [128, 413, 424]
    fraud_counts = torch.tensor([128, 413, 424], dtype=torch.float)
    fraud_weights = 1 / torch.sqrt(fraud_counts)
    # attack 子头: [reentrancy, logic error, price manipulation, access control, other] -> [50, 29, 22, 15, 245]
    attack_counts = torch.tensor([50, 29, 22, 15, 245], dtype=torch.float)
    attack_weights = 1 / torch.sqrt(attack_counts)
    criterion_l1 = nn.CrossEntropyLoss(weight=l1_weights, label_smoothing=label_smoothing)
    criterion_l2 = nn.CrossEntropyLoss(weight=l2_weights, label_smoothing=label_smoothing)
    criterion_attack = nn.CrossEntropyLoss(weight=attack_weights, label_smoothing=label_smoothing)
    criterion_fraud = nn.CrossEntropyLoss(weight=fraud_weights, label_smoothing=label_smoothing)
    return model, optimizer, criterion_l1, criterion_l2, criterion_attack, criterion_fraud
# model, optimizer, criterion = init_model(num_opcodes)
model, optimizer, criterion_l1, criterion_l2, criterion_attack, criterion_fraud = init_model(num_opcodes)
ATTACK_GLOBAL_LABELS = [4, 5, 6, 7, 8]
FRAUD_GLOBAL_LABELS = [1, 2, 3]
def to_attack_local(l3):
    local = torch.full_like(l3, -1)
    local[l3 == 4] = 0  # reentrancy
    local[l3 == 5] = 1  # logic error
    local[l3 == 6] = 2  # price manipulation
    local[l3 == 7] = 3  # access control
    local[l3 == 8] = 4  # other
    return local
def to_fraud_local(l3):
    local = torch.full_like(l3, -1)
    local[l3 == 1] = 0  # ponzi
    local[l3 == 2] = 1  # honeypot
    local[l3 == 3] = 2  # phishing
    return local
def build_global_l2_probs(preds):
    p1, p2, _, _ = preds
    p1_prob = F.softmax(p1, dim=1)
    p2_prob = F.softmax(p2, dim=1)
    batch_size = p1.shape[0]
    p2_global = torch.zeros(batch_size, 3, device=p1.device)
    # L2 全局定义: [benign, attack, fraud]
    p2_global[:, 0] = p1_prob[:, 0]
    p2_global[:, 1] = p1_prob[:, 1] * p2_prob[:, 0]
    p2_global[:, 2] = p1_prob[:, 1] * p2_prob[:, 1]
    return p2_global
def build_global_l3_probs(preds):
    p1, p2, p_attack, p_fraud = preds
    p1_prob = F.softmax(p1, dim=1)
    p2_prob = F.softmax(p2, dim=1)
    p_attack_prob = F.softmax(p_attack, dim=1)
    p_fraud_prob = F.softmax(p_fraud, dim=1)
    batch_size = p1.shape[0]
    p3_prob = torch.zeros(batch_size, OUT_DIM, device=p1.device)
    # benign -> 全局 L3 的 0 类
    p3_prob[:, 0] = p1_prob[:, 0]
    # fraud -> P(malicious) * P(fraud | malicious) * P(fraud leaf | fraud)
    fraud_gate = p1_prob[:, 1] * p2_prob[:, 1]
    p3_prob[:, 1:4] = fraud_gate.unsqueeze(1) * p_fraud_prob
    # attack -> P(malicious) * P(attack | malicious) * P(attack leaf | attack)
    attack_gate = p1_prob[:, 1] * p2_prob[:, 0]
    p3_prob[:, 4:9] = attack_gate.unsqueeze(1) * p_attack_prob
    return p3_prob
def hierarchical_predict(preds):
    p3_prob = build_global_l3_probs(preds)
    pred_l3 = p3_prob.argmax(dim=1)
    return pred_l3, p3_prob
def hierarchical_predict_l2(preds):
    p2_global = build_global_l2_probs(preds)
    return p2_global.argmax(dim=1)
def safe_metric_triplet(y_true, y_pred, average='weighted'):
    if len(y_true) == 0:
        return 0.0, 0.0, 0.0
    acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    return acc, precision, f1

def safe_macro_f1(y_true, y_pred):
    if len(y_true) == 0:
        return 0.0
    return f1_score(y_true, y_pred, average='macro', zero_division=0)
def local_l2_target_from_global(l2_tensor):
    # 原始全局 L2: benign=0, attack=1, fraud=2
    # 条件式局部 L2: attack=0, fraud=1
    local = torch.full_like(l2_tensor, -1)
    local[l2_tensor == 1] = 0
    local[l2_tensor == 2] = 1
    return local
def compute_hierarchical_inconsistency_count(pred_l1, pred_l2_global, pred_l3):
    inconsistency = 0
    for l1_hat, l2_hat, l3_hat in zip(pred_l1, pred_l2_global, pred_l3):
        # L3 -> implied parent labels
        if l3_hat == 0:
            implied_l1 = 0
            implied_l2 = 0
        elif l3_hat in [1, 2, 3]:
            implied_l1 = 1
            implied_l2 = 2
        else:
            implied_l1 = 1
            implied_l2 = 1
        # 一级与叶子层冲突
        if l1_hat != implied_l1:
            inconsistency += 1
            continue
        # malicious 条件下，二级与叶子层冲突
        if l1_hat == 1 and l2_hat != implied_l2:
            inconsistency += 1
    return inconsistency
    
# def compute_loss(preds, labels,
#                  w1=1.0, w2=0.7, w3=0.8, w_cons=0.05):
#     p1, p2, p3 = preds
#     l1, l2, l3 = labels
#     # ===== 基础 loss =====
#     loss1 = criterion_l1(p1, l1)
#     loss2 = criterion_l2(p2, l2)
#     loss3 = criterion_l3(p3, l3)
#     # ===== Consistency Loss =====
#     # softmax
#     p1_prob = F.softmax(p1, dim=1)
#     p2_prob = F.softmax(p2, dim=1)
#     p3_prob = F.softmax(p3, dim=1)
#     # ---------- L1 -> L3 ----------
#     benign_prob = p1_prob[:, 0]
#     malicious_prob = p1_prob[:, 1]
#     benign_l3_prob = p3_prob[:, 0]
#     malicious_l3_prob = 1 - benign_l3_prob
#     loss_cons_l1_l3 = F.mse_loss(benign_prob, benign_l3_prob) + \
#                      F.mse_loss(malicious_prob, malicious_l3_prob)
#     # ---------- L2 -> L3 ----------
#     # attack: [4,5,6,7]
#     attack_prob = p2_prob[:, 1]
#     fraud_prob = p2_prob[:, 2]
#     attack_l3_prob = p3_prob[:, [4,5,6,7,8]].sum(dim=1)
#     fraud_l3_prob = p3_prob[:, [1,2,3]].sum(dim=1)
#     loss_cons_l2_l3 = F.mse_loss(attack_prob, attack_l3_prob) + \
#                      F.mse_loss(fraud_prob, fraud_l3_prob)
#     loss_consistency = loss_cons_l1_l3 + loss_cons_l2_l3
#     # ===== 总 loss =====
#     total_loss = w1*loss1 + w2*loss2 + w3*loss3 + w_cons*loss_consistency
#     return total_loss
def compute_loss(preds, labels,
                 w1=1.0, w2=0.7, w3=0.8, w_cons=0.05):
    p1, p2, p_attack, p_fraud = preds
    l1, l2, l3 = labels
    # ===== Level 1: benign / malicious =====
    loss1 = criterion_l1(p1, l1)
    device = p1.device
    # ===== Level 2: attack / fraud（仅在 malicious 条件下训练） =====
    mal_mask = (l1 == 1)
    if mal_mask.any():
        # 原始标签: attack=1, fraud=2
        # 条件式局部标签: attack=0, fraud=1
        l2_malicious = torch.full_like(l2[mal_mask], -1)
        l2_malicious[l2[mal_mask] == 1] = 0
        l2_malicious[l2[mal_mask] == 2] = 1
        loss2 = criterion_l2(p2[mal_mask], l2_malicious)
    else:
        loss2 = torch.tensor(0.0, device=device)
    # ===== 条件式 L3 loss =====
    loss3_sum = torch.tensor(0.0, device=device)
    num_leaf_samples = 0
    attack_mask = (l2 == 1)
    if attack_mask.any():
        attack_targets = to_attack_local(l3[attack_mask])
        loss_attack = criterion_attack(p_attack[attack_mask], attack_targets)
        loss3_sum += loss_attack * attack_mask.sum()
        num_leaf_samples += attack_mask.sum().item()
    fraud_mask = (l2 == 2)
    if fraud_mask.any():
        fraud_targets = to_fraud_local(l3[fraud_mask])
        loss_fraud = criterion_fraud(p_fraud[fraud_mask], fraud_targets)
        loss3_sum += loss_fraud * fraud_mask.sum()
        num_leaf_samples += fraud_mask.sum().item()
    if num_leaf_samples > 0:
        loss3 = loss3_sum / num_leaf_samples
    else:
        loss3 = torch.tensor(0.0, device=device)
    # 结构上已经满足链式条件概率约束，不再额外施加 consistency loss
    total_loss = w1 * loss1 + w2 * loss2 + w3 * loss3
    return total_loss
def train_one_epoch(model, train_dataloader, optimizer):
    model.train()
    num_correct = 0
    num_tests = 0
    epoch_loss = 0
    for batched_graph, labels in train_dataloader:
        batched_graph = batched_graph
        labels = labels
        optimizer.zero_grad()
        pred = model(batched_graph, batched_graph.ndata['feat'], batched_graph.edata['etype'])
        
        l1 = labels[:, 0]
        l2 = labels[:, 1]
        l3 = labels[:, 2]
        loss = compute_loss(
            pred,
            (l1, l2, l3)
        )
        epoch_loss += loss.item()
        # 用L3算accuracy
        # pred_l3 = pred[2].argmax(1)
        pred_l3, _ = hierarchical_predict(pred)
        num_correct += (pred_l3 == l3).sum().item()
        num_tests += len(l3)
        loss.backward()
        optimizer.step()
    train_accuracy = num_correct / num_tests
    avg_loss = epoch_loss / len(train_dataloader)
    
    return avg_loss, train_accuracy, num_correct, num_tests
def evaluate(model, val_dataloader):
    model.eval()
    val_loss = 0
    val_correct = 0
    val_tests = 0
    with torch.no_grad():
        for batched_graph, labels in val_dataloader:
            batched_graph = batched_graph
            labels = labels
            pred = model(batched_graph, batched_graph.ndata['feat'], batched_graph.edata['etype'])
            l1 = labels[:, 0]
            l2 = labels[:, 1]
            l3 = labels[:, 2]
            loss = compute_loss(
                pred,
                (l1, l2, l3)
            )
            val_loss += loss.item()
            # pred_l3 = pred[2].argmax(1)
            pred_l3, _ = hierarchical_predict(pred)
            val_correct += (pred_l3 == l3).sum().item()
            val_tests += len(l3)
    
    val_loss /= len(val_dataloader)
    val_accuracy = val_correct / val_tests
    return val_loss, val_accuracy
def train_model(model, train_dataloader, val_dataloader, optimizer, num_epochs=EPOCH):
    train_losses, val_losses = [], []
    train_acc, val_acc = [], []
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    
    start_time = time.time()
    for epoch in range(num_epochs):
        train_loss, train_accuracy, num_correct, num_tests = train_one_epoch(model, train_dataloader, optimizer)
        val_loss, val_accuracy = evaluate(model, val_dataloader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_acc.append(train_accuracy)
        val_acc.append(val_accuracy)
        
        print(f"Epoch {epoch+1}: Train loss: {train_loss}, Train accuracy: {train_accuracy}, num_correct: {num_correct}, num_tests: {num_tests}")
        print(f"Epoch {epoch+1}: Validation loss: {val_loss}, Validation accuracy: {val_accuracy}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training time: {training_time}")
    
    return best_model_state, train_losses, val_losses, train_acc, val_acc, best_epoch, training_time
best_model_state, train_losses, val_losses, train_acc, val_acc, best_epoch, training_time = train_model(model, train_dataloader, val_dataloader, optimizer)
# Save the best model
torch.save(best_model_state, f'RGCN/model/trained_model/hie/best_model_hie_rgcn_{len(dataset)}_{best_epoch}-{EPOCH}.pt')
# def hierarchical_predict(preds):
#     p1, p2, p3 = preds
#     l1 = p1.argmax(1)
#     l2 = p2.argmax(1)
#     l3 = p3.argmax(1)
#     final = []
#     for i in range(len(l1)):
#         if l1[i] == 0:
#             final.append(0)
#         else:
#             final.append(1)
#     return torch.tensor(final)
L2_NAME = {
    0: "benign",
    1: "attack",
    2: "fraud"
}
L3_NAME = {
    0: "benign",
    1: "ponzi",
    2: "honeypot",
    3: "phishing",
    4: "reentrancy",
    5: "logic error",
    6: "price manipulation",
    7: "access control",
    8: "other"
}
def _normalize_meta_value(v):
    if v is None:
        return None
    if torch.is_tensor(v):
        if v.numel() == 1:
            return str(v.item())
        return str(v.tolist())
    return str(v)
def get_contract_creation_tx(dataset_obj, orig_idx, sample=None):
    """
    尽量从 dataset 或 sample 中取 contract_creation_tx。
    如果你的 ContractGraphDataset 中保存了 self.contract_creation_txs，
    这个函数会直接取到。
    """
    # 1) dataset 级别的 list / array
    for attr in ["contract_creation_txs", "contract_creation_txs", "contract_creation_tx_list", "tx_hashes", "contract_creation_tx"]:
        if hasattr(dataset_obj, attr):
            container = getattr(dataset_obj, attr)
            try:
                value = container[orig_idx]
                value = _normalize_meta_value(value)
                if value is not None:
                    return value
            except Exception:
                pass
    # 2) dataset 里若保存了 dataframe
    if hasattr(dataset_obj, "df"):
        df = getattr(dataset_obj, "df")
        if isinstance(df, pd.DataFrame) and "contract_creation_tx" in df.columns:
            try:
                return str(df.iloc[orig_idx]["contract_creation_tx"])
            except Exception:
                pass
    # 3) 从 sample 本身找
    if sample is not None:
        candidates = []
        if isinstance(sample, dict):
            candidates.append(sample)
            if "graph" in sample:
                candidates.append(sample["graph"])
        elif isinstance(sample, (tuple, list)):
            candidates.extend(sample)
        else:
            candidates.append(sample)
        for obj in candidates:
            if isinstance(obj, dict):
                for key in ["contract_creation_tx", "create_tx", "creationTx", "tx_hash"]:
                    if key in obj and obj[key] is not None:
                        return str(obj[key])
            else:
                for key in ["contract_creation_tx", "create_tx", "creationTx", "tx_hash"]:
                    if hasattr(obj, key):
                        value = getattr(obj, key)
                        if value is not None:
                            return str(value)
    return f"UNKNOWN_IDX_{orig_idx}"
# def test_model(model, test_dataloader):
#     model.eval()
#     num_correct = 0
#     num_tests = 0
#     all_preds, all_labels = [], []
#     test_start_time = time.time()
    
#     with torch.no_grad():
#         for batched_graph, labels in test_dataloader:
#             batched_graph = batched_graph
#             labels = labels
#             pred = model(batched_graph, batched_graph.ndata['feat'], batched_graph.edata['etype'])
            
#             # final_pred = hierarchical_predict(pred)
#             # gt = labels[:, 2]
#             # num_correct += (pred[2].argmax(1) == gt).sum().item()
#             # num_tests += len(gt)
#             # all_preds.extend(pred[2].argmax(1).tolist())
#             # all_labels.extend(gt.tolist())
#             gt = labels[:, 2]
#             pred_l3, _ = hierarchical_predict(pred)
#             num_correct += (pred_l3 == gt).sum().item()
#             num_tests += len(gt)
#             all_preds.extend(pred_l3.tolist())
#             all_labels.extend(gt.tolist())
    
#     test_end_time = time.time()
#     test_time = test_end_time - test_start_time
#     print(f"Test time: {test_time}")
    
#     test_accuracy = num_correct / num_tests
#     precision = precision_score(all_labels, all_preds, average='weighted')
#     recall = recall_score(all_labels, all_preds, average='weighted')
#     f1 = f1_score(all_labels, all_preds, average='weighted')
    
#     # Confusion Matrix
#     # TN FP
#     # FN TP
    
#     cm = confusion_matrix(
#     all_labels,
#     all_preds,
#     labels=[0,1,2,3,4,5,6,7,8]
#     )
#     print("Test Confusion Matrix:")
#     print(cm)
#     return test_accuracy, precision, recall, f1, cm, test_time
def test_model(model, test_dataloader, test_subset):
    model.eval()
    num_correct = 0
    num_tests = 0
    all_preds, all_labels = [], []
    attack_wrong_records = []
    test_start_time = time.time()
    # ===== 分层指标统计缓存 =====
    l1_true_all, l1_pred_all = [], []
    l2_true_all, l2_pred_all = [], []
    fraud_true_all, fraud_pred_all = [], []
    attack_true_all, attack_pred_all = [], []
    inconsistency_count = 0
    with torch.no_grad():
        for step, (batched_graph, labels) in enumerate(test_dataloader):
            pred = model(batched_graph, batched_graph.ndata['feat'], batched_graph.edata['etype'])
            gt_l1 = labels[:, 0]
            gt_l2 = labels[:, 1]
            gt_l3 = labels[:, 2]
            # 原始各层预测
            pred_l1 = pred[0].argmax(1)                  # 0 benign / 1 malicious
            pred_l2_global = hierarchical_predict_l2(pred)  # 0 benign / 1 attack / 2 fraud
            pred_l3, _ = hierarchical_predict(pred)
            num_correct += (pred_l3 == gt_l3).sum().item()
            num_tests += len(gt_l3)
            all_preds.extend(pred_l3.tolist())
            all_labels.extend(gt_l3.tolist())
            # ===== L1 指标 =====
            l1_true_all.extend(gt_l1.tolist())
            l1_pred_all.extend(pred_l1.tolist())
            # ===== L2 指标：仅在真实 malicious 条件下统计 =====
            mal_mask = (gt_l1 == 1)
            if mal_mask.any():
                gt_l2_local = local_l2_target_from_global(gt_l2[mal_mask])
                pred_l2_local = pred[1][mal_mask].argmax(1)  # 条件式二分类头 attack/fraud
                l2_true_all.extend(gt_l2_local.tolist())
                l2_pred_all.extend(pred_l2_local.tolist())
            # ===== Fraud branch 指标：仅在真实 fraud 叶子样本上统计 =====
            fraud_mask = (gt_l2 == 2)
            if fraud_mask.any():
                gt_fraud_local = to_fraud_local(gt_l3[fraud_mask])
                pred_fraud_local = pred[3][fraud_mask].argmax(1)
                fraud_true_all.extend(gt_fraud_local.tolist())
                fraud_pred_all.extend(pred_fraud_local.tolist())
            # ===== Adversarial branch 指标：仅在真实 attack 叶子样本上统计 =====
            attack_mask = (gt_l2 == 1)
            if attack_mask.any():
                gt_attack_local = to_attack_local(gt_l3[attack_mask])
                pred_attack_local = pred[2][attack_mask].argmax(1)
                attack_true_all.extend(gt_attack_local.tolist())
                attack_pred_all.extend(pred_attack_local.tolist())
            # ===== 层级一致性冲突统计 =====
            inconsistency_count += compute_hierarchical_inconsistency_count(
                pred_l1.tolist(), pred_l2_global.tolist(), pred_l3.tolist()
            )
            # 因为 test_dataloader 对 test_subset 是顺序遍历，所以 step 能对应回 test_subset[step]
            orig_idx = test_subset.indices[step]
            sample = test_subset[step]
            contract_creation_tx = get_contract_creation_tx(test_subset.dataset, orig_idx, sample)
            pred_l2_i = int(pred_l2_global.item())
            gt_l2_i = int(gt_l2.item())
            pred_l3_i = int(pred_l3.item())
            gt_l3_i = int(gt_l3.item())
            # ===== 只收集“预测为 attack 且预测错误”的项 =====
            if pred_l2_i == 1 and (pred_l2_i != gt_l2_i or pred_l3_i != gt_l3_i):
                error_type = "attack_false_positive" if gt_l2_i != 1 else "attack_subtype_error"
                attack_wrong_records.append({
                    "orig_idx": int(orig_idx),
                    "contract_creation_tx": contract_creation_tx,
                    "error_type": error_type,
                    "pred_l2": L2_NAME[pred_l2_i],
                    "true_l2": L2_NAME[gt_l2_i],
                    "pred_l3": L3_NAME[pred_l3_i],
                    "true_l3": L3_NAME[gt_l3_i]
                })
    test_end_time = time.time()
    test_time = test_end_time - test_start_time
    print(f"Test time: {test_time}")
    # ===== Global L3 指标 =====
    test_accuracy = num_correct / num_tests
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    # ===== 分层指标 =====
    l1_acc, l1_precision, l1_f1 = safe_metric_triplet(l1_true_all, l1_pred_all)
    l2_acc, l2_precision, l2_f1 = safe_metric_triplet(l2_true_all, l2_pred_all)
    fraud_acc, fraud_precision, fraud_f1 = safe_metric_triplet(fraud_true_all, fraud_pred_all)
    attack_acc, attack_precision, attack_f1 = safe_metric_triplet(attack_true_all, attack_pred_all)
    global_l3_acc, global_l3_precision, global_l3_f1 = safe_metric_triplet(all_labels, all_preds)
    fraud_macro_f1 = safe_macro_f1(fraud_true_all, fraud_pred_all)
    attack_macro_f1 = safe_macro_f1(attack_true_all, attack_pred_all)
    global_l3_macro_f1 = safe_macro_f1(all_labels, all_preds)
    overall_macro_f1 = global_l3_macro_f1
    inconsistency_rate = inconsistency_count / num_tests if num_tests > 0 else 0.0
    hierarchical_metrics = {
        "L1 Accuracy": l1_acc,
        "L1 Precision": l1_precision,
        "L1 F1-score": l1_f1,
        "L2 Accuracy": l2_acc,
        "L2 Precision": l2_precision,
        "L2 F1-score": l2_f1,
        "Fraud-branch Accuracy": fraud_acc,
        "Fraud-branch Precision": fraud_precision,
        "Fraud-branch F1-score": fraud_f1,
        "Fraud-branch Macro-F1": fraud_macro_f1,
        "Adversarial-branch Accuracy": attack_acc,
        "Adversarial-branch Precision": attack_precision,
        "Adversarial-branch F1-score": attack_f1,
        "Attack-branch Macro-F1": attack_macro_f1,
        "Global L3 Accuracy": global_l3_acc,
        "Global L3 Precision": global_l3_precision,
        "Global L3 F1-score": global_l3_f1,
        "Global L3 Macro-F1": global_l3_macro_f1,
        "Overall Macro-F1": overall_macro_f1,
        "Hierarchical Inconsistency Rate": inconsistency_rate,
    }
    cm = confusion_matrix(
        all_labels,
        all_preds,
        labels=[0, 1, 2, 3, 4, 5, 6, 7, 8]
    )
    print("Test Confusion Matrix:")
    print(cm)
    print("\n===== Hierarchical Metrics =====")
    for k, v in hierarchical_metrics.items():
        print(f"{k}: {v}")
    # ===== 输出 contract_creation_tx =====
    print("\nPredicted as attack but wrong:")
    if len(attack_wrong_records) == 0:
        print("No misclassified samples predicted as attack.")
    else:
        for rec in attack_wrong_records:
            print(
                f"contract_creation_tx={rec['contract_creation_tx']} | "
                f"type={rec['error_type']} | "
                f"true_l2={rec['true_l2']} | "
                f"true_l3={rec['true_l3']} | "
                f"pred_l3={rec['pred_l3']}"
            )
    # 只保存 contract_creation_tx，一行一个
    with open("attack_wrong_contract_creation_tx.txt", "w", encoding="utf-8") as f:
        for rec in attack_wrong_records:
            f.write(f"{rec['contract_creation_tx']}\n")
    # 保存完整记录，便于后续分析
    with open("attack_wrong_records.json", "w", encoding="utf-8") as f:
        json.dump(attack_wrong_records, f, ensure_ascii=False, indent=2)
    return test_accuracy, precision, recall, f1, cm, test_time, attack_wrong_records, hierarchical_metrics
# t_model, optimizer, criterion_l1, criterion_l2, criterion_l3 = init_model(num_opcodes)
t_model, optimizer, criterion_l1, criterion_l2, criterion_attack, criterion_fraud = init_model(num_opcodes)
t_model.load_state_dict(best_model_state)
# test_accuracy, precision, recall, f1, cm, test_time = test_model(t_model, test_dataloader)
test_accuracy, precision, recall, f1, cm, test_time, attack_wrong_records, hierarchical_metrics = test_model(
    t_model,
    test_dataloader,
    test_subset
)

# ================== 画 Loss & Acc ==================
def plot_loss_acc_curves(train_losses, val_losses, train_acc, val_acc, config, dataset_size):
    timestamp = int(time.time())
    exp_dir = f"RGCN/experiments/macro/hie_exp_{dataset_size}_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)
    # 平滑
    train_losses = list(np.hstack(pd.DataFrame(train_losses).rolling(5, min_periods=1).mean().values))
    val_losses = list(np.hstack(pd.DataFrame(val_losses).rolling(5, min_periods=1).mean().values))
    train_acc = list(np.hstack(pd.DataFrame(train_acc).rolling(5, min_periods=1).mean().values))
    val_acc = list(np.hstack(pd.DataFrame(val_acc).rolling(5, min_periods=1).mean().values))
    # 参数文字
    param_text = (
        f"LR={config['LR']}\n"
        f"EPOCH={config['EPOCH']}\n"
        f"H_DIM={config['H_DIM']}\n"
        f"OUT_DIM={config['OUT_DIM']}\n"
        f"BATCH_SIZE={config['BATCH_SIZE']}\n"
        f"DROP_OUT={config['DROP_OUT']}"
    )
    # ===== Loss =====
    plt.figure(figsize=(10, 8))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.text(0.65, 0.75, param_text, transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.6))
    loss_path = os.path.join(exp_dir, "loss_curve.png")
    plt.savefig(loss_path)
    plt.close()
    # ===== Accuracy =====
    plt.figure(figsize=(10, 8))
    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(val_acc, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy Curve')
    plt.legend()
    plt.text(0.65, 0.75, param_text, transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.6))
    acc_path = os.path.join(exp_dir, "accuracy_curve.png")
    plt.savefig(acc_path)
    plt.close()
    return exp_dir
# ================== 混淆矩阵 ==================
def plot_confusion_matrix(cm, config, exp_dir, class_names=None):
    cm = cm
    # 保存原始矩阵
    pd.DataFrame(cm).to_csv(os.path.join(exp_dir, "confusion_matrix.csv"), index=False)
    # ===== ✅ 保存 txt（你要的格式）=====
    txt_path = os.path.join(exp_dir, "confusion_matrix.txt")
    with open(txt_path, "w") as f:
        f.write("Test Confusion Matrix:\n")
        f.write(str(cm))   # 关键：直接写 numpy 格式
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    # plt.text(1.05, 0.5, param_text, transform=plt.gca().transAxes,
    #          bbox=dict(facecolor='white', alpha=0.6))
    cm_path = os.path.join(exp_dir, "confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()
    
# ================== 保存日志 ==================
def save_experiment_log(exp_dir, config, metrics):
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    with open(os.path.join(exp_dir, "metrics.txt"), "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
# ================== 使用示例 ==================
# ===== 参数 =====
config = {
    "LR": LR,
    "EPOCH": EPOCH,
    "H_DIM": H_DIM,
    "OUT_DIM": OUT_DIM,
    "BATCH_SIZE": BATCH_SIZE,
    "DROP_OUT": DROP_OUT
}
# ===== 指标 =====
metrics = {
    "Global L3 Accuracy": hierarchical_metrics["Global L3 Accuracy"],
    "Global L3 Precision": hierarchical_metrics["Global L3 Precision"],
    "Global L3 F1-score": hierarchical_metrics["Global L3 F1-score"],
    "L1 Accuracy": hierarchical_metrics["L1 Accuracy"],
    "L1 Precision": hierarchical_metrics["L1 Precision"],
    "L1 F1-score": hierarchical_metrics["L1 F1-score"],
    "L2 Accuracy": hierarchical_metrics["L2 Accuracy"],
    "L2 Precision": hierarchical_metrics["L2 Precision"],
    "L2 F1-score": hierarchical_metrics["L2 F1-score"],
    "Fraud-branch Accuracy": hierarchical_metrics["Fraud-branch Accuracy"],
    "Fraud-branch Precision": hierarchical_metrics["Fraud-branch Precision"],
    "Fraud-branch F1-score": hierarchical_metrics["Fraud-branch F1-score"],
    "Fraud-branch Macro-F1": hierarchical_metrics["Fraud-branch Macro-F1"],
    "Adversarial-branch Accuracy": hierarchical_metrics["Adversarial-branch Accuracy"],
    "Adversarial-branch Precision": hierarchical_metrics["Adversarial-branch Precision"],
    "Adversarial-branch F1-score": hierarchical_metrics["Adversarial-branch F1-score"],
    "Attack-branch Macro-F1": hierarchical_metrics["Attack-branch Macro-F1"],
    "Global L3 Macro-F1": hierarchical_metrics["Global L3 Macro-F1"],
    "Overall Macro-F1": hierarchical_metrics["Overall Macro-F1"],
    "Hierarchical Inconsistency Rate": hierarchical_metrics["Hierarchical Inconsistency Rate"],
    "Global L3 Recall (kept original)": recall,
    "test_time": test_time,
    "training_time": training_time
}
# ===== 控制台输出 =====
print("===== Experiment Config =====")
for k, v in config.items():
    print(f"{k}: {v}")
print("\n===== Metrics =====")
for k, v in metrics.items():
    print(f"{k}: {v}")
# ===== 绘制曲线 =====
exp_dir = plot_loss_acc_curves(
    train_losses,
    val_losses,
    train_acc,
    val_acc,
    config,
    len(dataset)
)
# ===== 收集预测（你测试阶段要有）=====
# 示例（替换为你的 test loop）
# all_preds = [...]
# all_labels = [...]
class_names = ["Benign", "Ponzi", "Honeypot", "Phishing", "Reentrancy", "Logic Error", "Price Manipulation", "Access Control","Others"]
# ===== 混淆矩阵 =====
plot_confusion_matrix(
    cm,
    config,
    exp_dir,
    class_names
)
# ===== 保存日志 =====
save_experiment_log(exp_dir, config, metrics)
print(f"\n所有结果已保存到: {exp_dir}")
