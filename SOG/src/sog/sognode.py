import typing as t

import networkx as nx

# import src.opcodes as opcodes
# from src.sog.sog_op import SOGOp, End, Project, BrRegion, Constant, Variable
# from src.tac_cfg import TACOp, TACArg

import opcodes as opcodes
from sog_op import SOGOp, End, Project, BrRegion, Constant, Variable
from tac_cfg import TACOp, TACArg


class DAGNode:
    def id(self):
        raise NotImplementedError

    def get_predecessors(self):
        raise NotImplementedError

    def get_successors(self):
        raise NotImplementedError

    def get_edge_type(self, pred_slot):
        raise NotImplementedError

    def get_feature_strs(self, opt):
        raise NotImplementedError


class SOGNode(DAGNode):
    id_cnt = 0
    n_uses_type = 4

    @staticmethod
    def clear_id_count():
        SOGNode.id_cnt = 0

    def __init__(self, op_name, opcode: opcodes.OpCode, args: t.List['TACArg'],
                 init_uses=0, pc=-1, block=None):
        self.id = SOGNode.id_cnt
        SOGNode.id_cnt += 1
        self.op_name = op_name
        self.opcode = opcode
        self.args = args
        self.pc = pc
        self.block = block

        self.uses = []
        # 0 - data, 1 - control, 2 - memory effect, 3 - other effect
        self.num_uses_per_type = [0] * SOGNode.n_uses_type
        self.num_uses_per_type[0] = init_uses
        self.defined_ops = []

    def __str__(self):
        return str(self.op_name)

    @classmethod
    def from_tac_op(cls, tac: 'TACOp'):
        return cls(tac, tac.opcode, tac.args, pc=tac.pc, block=tac.block)

    @classmethod
    def from_sog_op(cls, sog_op: 'SOGOp'):
        return cls(sog_op, sog_op.opcode, [])

    # def __init__(self, node):
    #     self.__init__(node.op, len(node.uses))
    #     for i in range(len(node.uses)):
    #         self.uses[i] = node.uses[i]
    #     n_uses_types = len(node.num_uses_per_type)
    #     self.num_uses_per_type = [0] * SOGNode.n_uses_type
    #     for i in range(n_uses_types):
    #         self.num_uses_per_type[i] = node.num_uses_per_type[i]
    #     self.defined_ops.extend(node.defined_ops)
    #     self.defined_node = node.defined_node
    #
    # def __init__(self, opc: int, init_uses: int):
    #     # TODO share baseop objects
    #     self.__init__(BaseOp(opc), init_uses)
    #     # assert opc != PcodeOp.MULTIEQUAL

    def id(self):
        return self.id

    def get_predecessors(self):
        return []

    def get_successors(self):
        return self.uses

    def get_edge_type(self, pred_slot):
        return 0

    def get_feature_strs(self, opt):
        return []

    def hash_code(self) -> int:
        return self.id

    def add_use(self, type, inp) -> None:
        assert SOGNode.n_uses_type >= type >= 0
        insert_at = 0
        for i in range(type + 1):
            insert_at += self.num_uses_per_type[i]
        self.uses.insert(insert_at, inp)
        self.num_uses_per_type[type] += 1

    def add_control_use(self, inp: 'SOGNode'):
        self.add_use(1, inp)
        # print(f"link:{inp.op_name}")
        # print(f"with {self.op_name}")
        # sog.add_edge(self, inp, type='control')

    @staticmethod
    def new_end():
        return SOGNode.from_sog_op(End())

    @classmethod
    def new_region(cls, op: 'TACOp') -> 'SOGNode':
        return SOGNode.from_tac_op(op)

    @classmethod
    def new_control_project(cls, input, offset) -> 'SOGNode':
        project = SOGNode.from_sog_op(Project(offset))
        project.add_control_use(input)
        project.add_control_use(SOGNode.new_constant(offset))
        return project

    @classmethod
    def new_br_region(cls, num_uses) -> 'SOGNode':
        return SOGNode.from_sog_op(BrRegion())

    @classmethod
    def new_constant(cls, c) -> 'SOGNode':
        return SOGNode.from_sog_op(Constant(c))

    @classmethod
    def new_variable(cls, name, value) -> 'SOGNode':
        return SOGNode.from_sog_op(Variable(name, value))


if __name__ == "__main__":
    node = SOGNode.new_end()
    print("Node ID:", node.id)
