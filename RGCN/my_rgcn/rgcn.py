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
import psutil
import os

LR = 0.0001
EPOCH = 1000
H_DIM = 16
OUT_DIM = 2
BATCH_SIZE = 16
DROP_OUT = 0.5

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")

with open('RGCN/processed_dataset/359/creation_1346_shuffled.pkl', 'rb') as f:
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

def count_labels(dataset, label=1):
    num_labels = 0
    for _, labels in dataset:
        num_labels += (labels == label).sum().item()
    return num_labels

num_labels_1 = count_labels(dataset)
print(f"Number of labels 1: {num_labels_1}")

def create_data_loaders(dataset, train_ratio=0.6, val_ratio=0.2):
    num_examples = len(dataset)
    num_train = int(num_examples * train_ratio)
    num_val = int(num_examples * val_ratio)
    # num_test = num_examples - num_train - num_val

    train_sampler = SubsetRandomSampler(torch.arange(num_train))
    val_sampler = SubsetRandomSampler(torch.arange(num_train, num_train + num_val))
    test_sampler = SubsetRandomSampler(torch.arange(num_train + num_val, num_examples))

    train_dataloader = GraphDataLoader(
        dataset, sampler=train_sampler, batch_size=BATCH_SIZE, drop_last=False
    )
    val_dataloader = GraphDataLoader(
        dataset, sampler=val_sampler, batch_size=BATCH_SIZE, drop_last=False
    )
    test_dataloader = GraphDataLoader(
        dataset, sampler=test_sampler, batch_size=1, drop_last=False
    )
    
    return train_dataloader, val_dataloader, test_dataloader

train_dataloader, val_dataloader, test_dataloader = create_data_loaders(dataset)

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

def init_model(num_opcodes, num_rels=3):
    model = RGCN(in_dim=num_opcodes, h_dim=H_DIM, out_dim=OUT_DIM, num_rels=num_rels)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    return model, optimizer, criterion

model, optimizer, criterion = init_model(num_opcodes)

def train_one_epoch(model, train_dataloader, optimizer, criterion):
    model.train()
    num_correct = 0
    num_tests = 0
    epoch_loss = 0
    for batched_graph, labels in train_dataloader:
        batched_graph = batched_graph
        labels = labels
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
    
    return avg_loss, train_accuracy, num_correct, num_tests

def evaluate(model, val_dataloader, criterion):
    model.eval()
    val_loss = 0
    val_correct = 0
    val_tests = 0
    with torch.no_grad():
        for batched_graph, labels in val_dataloader:
            batched_graph = batched_graph
            labels = labels
            pred = model(batched_graph, batched_graph.ndata['feat'], batched_graph.edata['etype'])
            val_correct += (pred.argmax(1) == labels).sum().item()
            val_tests += len(labels)
            val_loss += criterion(pred, labels).item()
    
    val_loss /= len(val_dataloader)
    val_accuracy = val_correct / val_tests
    return val_loss, val_accuracy

def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024

def train_model(model, train_dataloader, val_dataloader, optimizer, criterion, num_epochs=EPOCH):
    train_losses, val_losses = [], []
    train_acc, val_acc = [], []
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    
    start_time = time.time()

    for epoch in range(num_epochs):
        train_loss, train_accuracy, num_correct, num_tests = train_one_epoch(model, train_dataloader, optimizer, criterion)
        val_loss, val_accuracy = evaluate(model, val_dataloader, criterion)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_acc.append(train_accuracy)
        val_acc.append(val_accuracy)
        
        print(f"Epoch {epoch+1}: Train loss: {train_loss}, Train accuracy: {train_accuracy}, num_correct: {num_correct}, num_tests: {num_tests}")
        print(f"Epoch {epoch+1}: Validation loss: {val_loss}, Validation accuracy: {val_accuracy}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            best_epoch = epoch
    
    end_time = time.time()
    print(f"Training time: {end_time - start_time}")
    
    return best_model_state, train_losses, val_losses, train_acc, val_acc, best_epoch

best_model_state, train_losses, val_losses, train_acc, val_acc, best_epoch = train_model(model, train_dataloader, val_dataloader, optimizer, criterion)

# Save the best model
torch.save(best_model_state, f'RGCN/model/trained_model/ponzi_rgcn_{len(dataset)}_{best_epoch}-{EPOCH}-{time.time()}.pt')

def test_model(model, test_dataloader):
    model.eval()
    num_correct = 0
    num_tests = 0
    all_preds, all_labels = [], []
    failed_samples = []
    test_start_time = time.time()
    wrong = []
    
    with torch.no_grad():
        for batched_graph, labels in test_dataloader:
            batched_graph = batched_graph
            labels = labels
            pred = model(batched_graph, batched_graph.ndata['feat'], batched_graph.edata['etype'])

            if pred.argmax(1) != labels:
                wrong.append([pred.argmax(1), labels, batched_graph])

            num_correct += (pred.argmax(1) == labels).sum().item()
            num_tests += len(labels)
            all_preds.extend(pred.argmax(1).tolist())
            all_labels.extend(labels.tolist())

            # for i in range(len(labels)):
            #     if pred.argmax(1)[i] != labels[i]:
            #         failed_samples.append({
            #             'graph': batched_graph,  
            #             'label': labels[i].item(),
            #             'predicted': pred.argmax(1)[i].item(),
            #             'index': batched_graph.batch_idx[i].item(),  
            #             'original_data': dataset[batched_graph.batch_idx[i].item()]  
            #             })
    
    test_end_time = time.time()
    print(f"Test time: {test_end_time - test_start_time}")
    
    test_accuracy = num_correct / num_tests
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    
    # Confusion Matrix
    # TN FP
    # FN TP
    cm = confusion_matrix(all_labels, all_preds)
    print("Test Confusion Matrix:")
    print(cm)
    print(wrong)

    # print(f"\nNumber of failed predictions: {len(failed_samples)}")
    # if len(failed_samples) > 0:
    #     print("\nSome failed predictions:")
    #     for failed_sample in failed_samples:
    #         print(f"Index: {failed_sample['index']}, Label: {failed_sample['label']}, Predicted: {failed_sample['predicted']}")
    #         print(f"Original data: {failed_sample['original_data']}")

    return test_accuracy, precision, recall, f1
t_model, optimizer, criterion = init_model(num_opcodes)
t_model.load_state_dict(best_model_state)

test_accuracy, precision, recall, f1 = test_model(t_model, test_dataloader)
print(f"Test accuracy: {test_accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {f1}")

def plot_loss_acc_curves(train_losses, val_losses, train_acc, val_acc):
    plt.figure(figsize=(10, 8))
    df = pd.DataFrame(train_losses)
    train_losses = list(np.hstack(df.rolling(5, min_periods=1).mean().values))
    df1 = pd.DataFrame(val_losses)
    val_losses = list(np.hstack(df1.rolling(5, min_periods=1).mean().values))
    
    plt.plot(range(len(train_losses)), train_losses, label='Train Loss')
    plt.plot(range(len(val_losses)), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.savefig(f"RGCN/losses/ponzi/ponzi_loss_curve_{len(dataset)}_{time.time()}.png")
    
    plt.figure(figsize=(10, 8))
    df2 = pd.DataFrame(train_acc)
    train_acc = list(np.hstack(df2.rolling(5, min_periods=1).mean().values))
    df3 = pd.DataFrame(val_acc)
    val_acc = list(np.hstack(df3.rolling(5, min_periods=1).mean().values))
    
    plt.plot(range(len(train_acc)), train_acc, label='Train Accuracy')
    plt.plot(range(len(val_acc)), val_acc, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy Curve')
    plt.legend()
    plt.savefig(f"RGCN/accs/ponzi/ponzi_accuracy_curve_{len(dataset)}_{time.time()}.png")

plot_loss_acc_curves(train_losses, val_losses, train_acc, val_acc)
