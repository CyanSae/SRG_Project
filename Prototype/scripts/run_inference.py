"""独立的 RGCN 推理脚本。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import RelGraphConv


OPCODES = [
    "CONST", "JUMPDEST", "ADD", "JUMP", "MSTORE", "JUMPI", "AND", "MLOAD", "ISZERO", "SUB", "REVERT", "SHL",
    "EQ", "SLOAD", "SHA3", "LT", "MUL", "RETURNDATASIZE", "CALLDATALOAD", "GT", "DIV", "CALLVALUE", "EXP", "SSTORE",
    "NOT", "CALLDATASIZE", "RETURN", "CALLER", "SLT", "RETURNDATACOPY", "OR", "LOG", "GAS", "EXTCODESIZE", "CODECOPY",
    "STOP", "CALL", "ADDRESS", "INVALID", "CALLDATACOPY", "STATICCALL", "SHR", "GASPRICE", "TIMESTAMP", "DELEGATECALL",
    "GASLIMIT", "NOP", "ADDMOD", "SIGNEXTEND", "BALANCE", "MOD", "SMOD", "SGT", "MSTORE8", "ORIGIN", "BYTE", "NUMBER",
    "MISSING", "SDIV", "CREATE2", "CALLCODE", "CREATE", "MULMOD", "EXTCODEHASH", "COINBASE", "SELFDESTRUCT", "CODESIZE",
    "XOR", "BLOCKHASH", "DIFFICULTY", "SAR", "EXTCODECOPY", "MSIZE", "PC",
]

SUBTYPE_LABELS = {
    0: ("benign", "正常合约"),
    1: ("ponzi", "庞氏骗局"),
    2: ("honeypot", "蜜罐合约"),
    3: ("phishing", "钓鱼欺诈"),
    4: ("reentrancy", "重入攻击"),
    5: ("logic error", "逻辑错误"),
    6: ("price manipulation", "价格操纵"),
    7: ("access control", "访问控制问题"),
    8: ("other", "其他对抗性类型"),
}

CATEGORY_LABELS = {
    0: ("benign", "正常合约"),
    1: ("attack", "对抗性合约"),
    2: ("fraud", "欺诈合约"),
}

BINARY_LABELS = {
    0: ("benign", "正常"),
    1: ("malicious", "恶意"),
}


class HierarchicalRGCN(nn.Module):
    """与训练脚本保持一致的层次化 RGCN 模型。"""

    def __init__(self, in_dim: int, h_dim: int, num_rels: int) -> None:
        super().__init__()
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
            h_dim,
            num_rels,
            regularizer="basis",
            num_bases=num_rels,
            self_loop=False,
        )
        self.dropout = nn.Dropout(0.3)
        self.fc_l1 = nn.Linear(h_dim, 2)
        self.fc_l2 = nn.Linear(h_dim, 2)
        self.fc_attack = nn.Linear(h_dim, 5)
        self.fc_fraud = nn.Linear(h_dim, 3)

    def forward(self, graph: dgl.DGLGraph, feat: torch.Tensor, etype: torch.Tensor):
        """执行前向推理。"""

        hidden = F.relu(self.conv1(graph, feat, etype))
        hidden = self.conv2(graph, hidden, etype)
        hidden = self.dropout(hidden)
        with graph.local_scope():
            graph.ndata["h"] = hidden
            pooled = dgl.mean_nodes(graph, "h")
        return (
            self.fc_l1(pooled),
            self.fc_l2(pooled),
            self.fc_attack(pooled),
            self.fc_fraud(pooled),
        )


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="对单个 SRG JSON 执行 RGCN 推理。")
    parser.add_argument("--graph-json", required=True, help="SRG JSON 文件路径。")
    parser.add_argument("--model-path", required=True, help="训练好的模型权重路径。")
    parser.add_argument("--output", required=True, help="推理结果输出路径。")
    return parser.parse_args()


def load_graph(graph_json_path: Path) -> dgl.DGLGraph:
    """将 SRG JSON 转换为 DGL 图。"""

    with graph_json_path.open("r", encoding="utf-8") as file:
        graph_data = json.load(file)

    nodes = graph_data.get("nodes", {})
    edges = graph_data.get("edges", [])
    if not nodes or not edges:
        raise RuntimeError("SRG 图为空，无法执行检测。")

    node_ids = list(nodes.keys())
    node_map = {int(node_id): index for index, node_id in enumerate(node_ids)}
    src_nodes = [node_map[int(edge[0])] for edge in edges]
    dst_nodes = [node_map[int(edge[1])] for edge in edges]
    graph = dgl.graph((src_nodes, dst_nodes))

    opcode_to_feature = {opcode: index for index, opcode in enumerate(OPCODES)}
    feature_indexes = [opcode_to_feature.get(nodes[node_id], 0) for node_id in node_ids]
    graph.ndata["feat"] = F.one_hot(torch.tensor(feature_indexes), num_classes=len(OPCODES)).float()
    graph.edata["etype"] = torch.tensor([int(edge[2]) for edge in edges]).long()
    return graph


def build_global_l2_probs(preds: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """构造全局二级标签概率。"""

    pred_l1, pred_l2, _, _ = preds
    prob_l1 = F.softmax(pred_l1, dim=1)
    prob_l2 = F.softmax(pred_l2, dim=1)
    result = torch.zeros(pred_l1.shape[0], 3)
    result[:, 0] = prob_l1[:, 0]
    result[:, 1] = prob_l1[:, 1] * prob_l2[:, 0]
    result[:, 2] = prob_l1[:, 1] * prob_l2[:, 1]
    return result


def build_global_l3_probs(preds: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """构造全局叶子标签概率。"""

    pred_l1, pred_l2, pred_attack, pred_fraud = preds
    prob_l1 = F.softmax(pred_l1, dim=1)
    prob_l2 = F.softmax(pred_l2, dim=1)
    prob_attack = F.softmax(pred_attack, dim=1)
    prob_fraud = F.softmax(pred_fraud, dim=1)
    result = torch.zeros(pred_l1.shape[0], 9)
    result[:, 0] = prob_l1[:, 0]
    fraud_gate = prob_l1[:, 1] * prob_l2[:, 1]
    attack_gate = prob_l1[:, 1] * prob_l2[:, 0]
    result[:, 1:4] = fraud_gate.unsqueeze(1) * prob_fraud
    result[:, 4:9] = attack_gate.unsqueeze(1) * prob_attack
    return result


def to_prob_dict(probabilities: torch.Tensor, labels: dict[int, tuple[str, str]]) -> list[dict[str, object]]:
    """将概率向量转换为便于前端展示的结构。"""

    rows = []
    for index, value in enumerate(probabilities.tolist()):
        code, name_zh = labels[index]
        rows.append(
            {
                "index": index,
                "code": code,
                "label_zh": name_zh,
                "confidence": round(float(value), 6),
            }
        )
    rows.sort(key=lambda item: item["confidence"], reverse=True)
    return rows


def build_summary(label_zh: str, confidence: float, prefix: str) -> str:
    """构造中文摘要。"""

    return f"{prefix}结果为“{label_zh}”，置信度约为 {confidence:.2%}。"


def main() -> int:
    """脚本主入口。"""

    args = parse_args()
    graph_json_path = Path(args.graph_json)
    model_path = Path(args.model_path)
    output_path = Path(args.output)

    graph = load_graph(graph_json_path)
    model = HierarchicalRGCN(in_dim=len(OPCODES), h_dim=32, num_rels=4)
    try:
        # 优先使用更安全的权重加载方式。
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        preds = model(graph, graph.ndata["feat"], graph.edata["etype"])
        binary_probs = F.softmax(preds[0], dim=1)[0]
        category_probs = build_global_l2_probs(preds)[0]
        subtype_probs = build_global_l3_probs(preds)[0]

    binary_index = int(binary_probs.argmax().item())
    category_index = int(category_probs.argmax().item())
    subtype_index = int(subtype_probs.argmax().item())

    binary_code, binary_label_zh = BINARY_LABELS[binary_index]
    category_code, category_label_zh = CATEGORY_LABELS[category_index]
    subtype_code, subtype_label_zh = SUBTYPE_LABELS[subtype_index]

    result = {
        "binary": {
            "index": binary_index,
            "code": binary_code,
            "label_zh": binary_label_zh,
            "confidence": round(float(binary_probs[binary_index].item()), 6),
            "summary": build_summary(binary_label_zh, float(binary_probs[binary_index].item()), "一级检测"),
            "ranking": to_prob_dict(binary_probs, BINARY_LABELS),
        },
        "category": {
            "index": category_index,
            "code": category_code,
            "label_zh": category_label_zh,
            "confidence": round(float(category_probs[category_index].item()), 6),
            "summary": build_summary(category_label_zh, float(category_probs[category_index].item()), "二级检测"),
            "ranking": to_prob_dict(category_probs, CATEGORY_LABELS),
        },
        "subtype": {
            "index": subtype_index,
            "code": subtype_code,
            "label_zh": subtype_label_zh,
            "confidence": round(float(subtype_probs[subtype_index].item()), 6),
            "summary": build_summary(subtype_label_zh, float(subtype_probs[subtype_index].item()), "三级检测"),
            "ranking": to_prob_dict(subtype_probs, SUBTYPE_LABELS),
        },
        "graphStats": {
            "nodeCount": int(graph.num_nodes()),
            "edgeCount": int(graph.num_edges()),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
