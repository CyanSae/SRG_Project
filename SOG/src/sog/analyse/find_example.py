import os
import json
from collections import defaultdict

directory = '/home/sandra/projects/DATA/SOG_SET/2367'

for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as f:
                graph_data = json.load(f)
                node_count = graph_data.get("node_count", {})
                edge_count = graph_data.get("edge_count", [])
                if node_count == 1256 and edge_count == 1087:
                     print(filename)