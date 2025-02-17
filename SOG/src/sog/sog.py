class SOG:

    def __init__(self, end_node):
        self.end = end_node

    def root(self):
        return self.end

    def workroots(self):
        roots_list = [self.end]
        return roots_list
