from collections import deque

import networkx as nx
from matplotlib import pyplot as plt

import sys
sys.path.append('src')
import json
# import src.dataflow as dataflow
# import src.opcodes as opcodes
# import src.sog.sognode as sognode
# import src.tac_cfg as tac_cfg
import dataflow
import tac_cfg
import opcodes
import sog.sognode as sognode


class SOGBuilder:
    def __init__(self, cfg: tac_cfg.TACGraph):
        self.work_list = deque()
        self.processed = set()
        self.end = sognode.SOGNode.new_end()
        self.regions = {}
        self.sog = nx.DiGraph()
        self.cfg = cfg
        self.blocks = self.cfg.blocks
        self.define = {}
        self.use = {}

    def build(self):
        # Step 1: Get dominance frontiers
        # dom_frontiers = self.cfg.get_dominance_frontiers()

        # Step 2: Build regions and insert PHI nodes
        # self.sog.clear()
        # print(f"initial:{self.sog}")
        # self.build_regions()
        # self.insert_phi_nodes(dom_frontiers)  # Integrate PHI node insertion here

        # Step 3: Build control, data, and effect sub graphs
        self.build_control_subgraph()
        self.build_data_subgraph()
        self.build_effect_subgraph()

        return self.sog

        # Step 1. Get dominator frontiers
        # # dom = self.cfg.dominators()
        # children_list_in_dt = self.cfg.get_children_list_in_dt()
        #
        # Step 2. Construct regions and insert PHI nodes
        # self.build_regions()
        # self.prepare_phi()
        # self.insert_phi_nodes(dom)
        #
        # # Step 3. Construct Sea of Nodes.
        # return self.builds(children_list_in_dt)

    # def insert_phi_nodes(self, dom):
    #     pass  # Logic to insert PHI nodes
    #
    # def build_regions(self):
    #     pass  # Logic to build regions
    #
    # def prepare_phi(self):
    #     pass  # Logic to prepare PHI
    #
    # def builds(self, bfs_order_tree):
    #     return None  # Logic to build SOG
    #
    # def build_one_block(self, bl):
    #     pass  # Logic to build one block

    def build_regions(self):
        fallthrough = {b.ident(): -1 for b in self.blocks}
        for n in self.blocks:
            last = n.last_op
            # 创建 SOGNode 来表示该基本块的结束操作
            control_node = sognode.SOGNode.new_region(last)
            if last is not None and last.opcode == opcodes.JUMPI and len(n.succs) == 2:
                fallthrough[n.ident()] = 0

            self.regions[n.ident()] = control_node  # 将控制节点添加到区域列表
            # 处理区域控制输入
        for bl in self.blocks:
            bl_region = self.regions[bl.ident()]
            for pred in bl.preds:
                prid = pred.ident()
                if fallthrough[prid] == -1:
                    bl_region.add_control_use(self.regions[prid])
                    # self.sog.add_edge(bl_region, self.regions[prid])
                else:
                    proj_idx = 0 if pred.succs.index(bl) == fallthrough[prid] else 1
                    pr_region = self.regions[prid]
                    cproj = sognode.SOGNode.new_control_project(pr_region, proj_idx)
                    bl_region.add_control_use(cproj)

        start = sognode.SOGNode.new_br_region(0)
        first_key = next(iter(self.regions))
        self.regions[first_key].add_control_use(start)
        # for r in self.regions.values():
        #     print(f"id:{r.id},node:{r}")
        #     for u in r.uses:
        #         print(f"uses:{u},id:{u.id}")

        # add:BR in 0x31: JUMPI 0x37 V7
        # self.sog.add_edge(self.regions[first_key], start)
        # print(fallthrough)
        # for sogn in self.regions.values():
        #     print(sogn.op_name)

    def insert_phi_nodes(self, domfrontiers):
        pass

    def build_control_subgraph(self):
        """
        Constructs the control subgraph by adding control flow edges
        between branches and their targets.
        """
        for block in self.blocks:
            last_op = block.last_op
            # if not last_op.is_branch():
            # Insert a dummy branch instruction if needed
            # dummy_branch = self.create_dummy_branch(block)
            # block.add_instruction(dummy_branch)

            for succ in block.succs:
                self.sog.add_edge(sognode.SOGNode.from_tac_op(succ.first_op).op_name,
                                  sognode.SOGNode.from_tac_op(last_op).op_name,
                                  type='control', color='blue')

            first_op = block.first_op
            if first_op.opcode == opcodes.JUMPDEST and len(block.tac_ops) >= 2:
                next_op = block.tac_ops[1]
                self.sog.add_edge(next_op, first_op, type='control', color='blue')

    def build_data_subgraph(self):
        """
        Constructs the data subgraph by analyzing def-use relations
        between instructions.
        """
        # Mapping from variable names to the addresses they were defined at.
        # define = {}
        # Mapping from variable names to the addresses they were used at.
        # use = {}
        # Mapping from variable names to their possible values.
        value = []
        stack = {}
        for block in self.blocks:
            for var in block.entry_stack:
                if not var.def_sites.is_const and var.def_sites.is_finite:
                    name = block.ident() + ":" + var.name
                    for loc in var.def_sites:
                        tacop = self.cfg.get_ops_by_pc(loc.pc)
                        # def_node = sognode.SOGNode.from_tac_op(tacop)
                        # def_node = sognode.SOGNode.new_variable(name, -1)
                        stack[name] = [hex(loc.pc)]
            for op in block.tac_ops:
                if isinstance(op, tac_cfg.TACAssignOp):
                    def_node = sognode.SOGNode.from_tac_op(op)
                    # def_node = sognode.SOGNode.new_variable(op.lhs.name, op.lhs.values)
                    self.define[op.lhs.name] = [hex(op.pc), def_node]
                    if op.lhs.values.is_finite:
                        for val in op.lhs.values:
                            value.append((op.lhs, hex(val)))
        for block in self.blocks:
            for op in block.tac_ops:
                if op.opcode.has_use_sites():
                    # The args constitute use sites.
                    for i, arg in enumerate(op.args):
                        name = arg.value.name
                        if not arg.value.def_sites.is_const:
                            # Argument is a stack variable, and therefore needs to be
                            # prepended with the block id.
                            name = block.ident() + ":" + name
                        # relation format: use(Var, PC, ArgIndex)
                        # use_node = sognode.SOGNode.new_variable(name, value)
                        use_node = sognode.SOGNode.from_tac_op(op)
                        self.use[name] = [hex(op.pc), i + 1, use_node]
                        if self.define.get(name) is not None:
                            self.sog.add_edge(use_node.op_name, self.define[name][1].op_name,
                                              type='data', color='green')
                # self.sog.add_node(tac_op.pc, type='instruction', opcode=tac_op.opcode)

        # print("stack:")
        # print(self.use)
        # print(self.define)
        # if use.get(name) is not None:
        #     self.sog.add_edge(use[name][2], def_node, type='data')

    def build_effect_subgraph(self):
        """
        Constructs the effect subgraph by analyzing memory and store effects.
        """
        def_sites = {}
        use_sites = {}
        use_sites_1 = {}
        # use_ops=[]
        # def_ops=[]
        for block in self.blocks:
            for op in block.tac_ops:
                if op.opcode in [opcodes.MSTORE, opcodes.MSTORE8, opcodes.SSTORE]:
                    effect_slot = op.args[0].__str__()
                    # if def_sites.get(effect_slot) is None:
                    #     def_sites[effect_slot] = []
                    def_sites[effect_slot] = op
                if op.opcode in [opcodes.SLOAD, opcodes.MLOAD]:
                    if self.use.get(op.lhs.name) is not None:
                        use_site = self.use[op.lhs.name]

                        if use_sites.get(hex(op.pc)) is None:
                            use_sites[hex(op.pc)] = []
                        use_sites[hex(op.pc)].append(use_site)
                        def_op = self.cfg.get_ops_by_pc(int(use_site[0], 16))
                        self.sog.add_edge(def_op[0], op,
                                          type='effect', color='red')
                        # use_sites_1[self.cfg.get_ops_by_pc(op.pc)] = self.cfg.get_ops_by_pc(int(use_site[0], 16))
                    effect_slot = op.args[0].__str__()
                    if def_sites.get(effect_slot) is not None:
                        self.sog.add_edge(op, def_sites.get(effect_slot), type='effect', color='red')

        # print(use_ops)
        # print(def_ops)
        # print(use_sites)
        # print(def_sites)


def save_to_json(adjacency_view):
    edge_type_map = {"data": 0, "control": 1, "effect": 2}

    # 初始化数据结构
    nodes = {}
    edges_list = []

    # 解析边并填充数据结构
    for node, neighbors in adjacency_view:
        for neighbor, edge_attrs in neighbors.items():
            node.name = node.opcode.__str__()
            neighbor.name = neighbor.opcode.__str__()
            if node.opcode == opcodes.CONST:
                node.name = f"CONST({node.args[0]})"
            if neighbor.opcode == opcodes.CONST:
                neighbor.name = f"CONST({neighbor.args[0]})"
            edge_type = edge_attrs.get('type', 'unknown')
            if node.pc not in nodes:
                nodes[node.pc] = node.name
            if neighbor.pc not in nodes:
                nodes[neighbor.pc] = neighbor.name
            edge_type_index = edge_type_map[edge_type]
            edges_list.append([node.pc, neighbor.pc, edge_type_index])
            # print(f"{edge_type} edge: {node} ----- {neighbor}")
            print(f"{edge_type} edge: ({hex(node.pc)}){node.name} ----- ({hex(neighbor.pc)}){neighbor.name}")
    # print(len(sog.nodes()))
    # print(len(sog.edges()))

    # 构建JSON对象
    graph_json = {
        "nodes": nodes,
        "edges": edges_list,
        "node_count": len(nodes),
        "edge_count": len(edges_list)
    }

    # 将JSON对象写入文件
    # with open('D:\Codes\Projects\SOG\data\output\sog_graph.json', 'w') as f:
    #     json.dump(graph_json, f, indent=4)
    #
    # print("Graph has been successfully converted to JSON format and saved to 'sog_graph.json'.")


# Example usage
if __name__ == "__main__":
    print("test:\n")
    file_path = "D:/Codes/Projects/SOG/examples/basic.hex"
    f = open(file_path, 'r')
    cfg = tac_cfg.TACGraph.from_bytecode(f.read())
    dataflow.analyse_graph(cfg)
    dataflow.analyse_graph(cfg)

    builder = SOGBuilder(cfg)
    sog = builder.build()
    print("SOG built:")
    # nx.draw(sog, with_labels=True)
    # # plt.show()
    # for node in sog.nodes():
    #     print(f"node:{node}")
    adjacency_view = sog.adjacency()
    save_to_json(adjacency_view)
    # print(sog.nodes)
    # for edge in sog.edges():
    #     print(f"edge: {edge[0]}------{edge[1]}")

    nx.draw(sog, pos=nx.random_layout(sog), with_labels=True, edge_color=['red', 'blue', 'green'])
    # 'blue', 'green', pos=nx.circular_layout(sog),
    plt.show()
