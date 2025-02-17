# import src.opcodes as opcodes
import opcodes as opcodes


class SOGOp:
    def __init__(self, opcode):
        """
        Args:
          opcode: code
        """
        self.opcode = opcode

    # @staticmethod
    # def project(out_size):
    #     """
    #     Simulates the behavior of the Project class in a functional way.
    #
    #     :param out_size: The output size for the projection operation.
    #     :return: A string representing the projection operation.
    #     """
    #     return f"PROJ({out_size})"


class Project(SOGOp):
    def __init__(self, out_size):
        super().__init__(opcodes.Project)
        self.out_size = out_size

    def __str__(self):
        return f"PROJ({self.out_size})"


class Variable(SOGOp):
    def __init__(self, name, value):
        super().__init__(opcodes.Variable)
        self.name = name
        self.value = value

    def __str__(self):
        return f"{self.name}"


class BrRegion(SOGOp):
    def __init__(self):
        super().__init__(opcodes.BrRegion)

    def __str__(self):
        return "BR"


class Constant(SOGOp):
    def __init__(self, constant):
        super().__init__(opcodes.CONST)
        self.constant = constant

    def __str__(self):
        return f"C({self.constant})"


class End(SOGOp):
    def __init__(self):
        super().__init__(opcodes.End)
