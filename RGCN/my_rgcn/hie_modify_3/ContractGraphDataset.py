from dgl.data import DGLDataset

class ContractGraphDataset(DGLDataset):
    def __init__(self, json_dir):
        self.json_dir = json_dir
        super().__init__(name='contract_graph')

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        return len(self.graphs)