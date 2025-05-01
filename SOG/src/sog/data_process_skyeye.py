import pandas as pd
import os
import json
from sog_builder_connect import SOGBuilder
import dataflow
import tac_cfg
import opcodes
# import src.dataflow as dataflow
# import src.tac_cfg as tac_cfg
# import src.opcodes as opcodes

import concurrent.futures
import time


# def save_to_json(data, file_path):
#     with open(file_path, 'w') as f:
#         json.dump(data, f, indent=4)


def save_to_json(adjacency_view, file_path):
    edge_type_map = {"data": 0, "control": 1, "effect": 2, "connection": 3}

    nodes = {}
    edges_list = []

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
    #         print(f"{edge_type} edge: ({hex(node.pc)}){node.name} ----- ({hex(neighbor.pc)}){neighbor.name}")
    # print(len(sog.nodes()))
    # print(len(sog.edges()))

    graph_json = {
        "nodes": nodes,
        "edges": edges_list,
        "node_count": len(nodes),
        "edge_count": len(edges_list)
    }

    with open(file_path, 'w') as f:
        json.dump(graph_json, f, indent=4)

    # print(f"Graph has been successfully converted to JSON format and saved to {file_path}.")


class TimeoutException(Exception):
    pass


def process_with_timeout(data, timeout=30):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(analyse, data)
        try:
            result = future.result(timeout=timeout)
            return result
        except concurrent.futures.TimeoutError:
            print(f"Processing time exceeded the limit of {timeout} seconds for data.")
            # 处理下一个


def analyse(cfg):
    dataflow.analyse_graph(cfg)


def process_bytecode(bytecode, output_dir, contract_address):
    if bytecode != "" and bytecode != "0x":
        cfg = tac_cfg.TACGraph.from_bytecode(bytecode)
        process_with_timeout(cfg)
        # dataflow.analyse_graph(cfg)

        builder = SOGBuilder(cfg)
        sog = builder.build()

        adjacency_view = sog.adjacency()
        json_file_path = os.path.join(output_dir, f"{contract_address}.json")
        save_to_json(adjacency_view, json_file_path)
        # print(f"{contract_address} done")


def main(csv_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_file)
    results = []
    for index, (i, row) in enumerate(df.iterrows(), start=171):
        if i < 171:
            continue
        start_time = time.time()
        contract_address = row['adversarial contracts']

        # bytecode = row['bytecode']
        directory = '/home/sandra/project/SOG/src/174/adversarial_contracts_bytecode'
        file_path = os.path.join(directory, f"{contract_address}.txt")

        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            continue

        with open(file_path, 'r') as file:
            bytecode = file.read().strip()

        process_bytecode(bytecode, output_dir, contract_address)
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"{contract_address} done using {processing_time}")
        results.append((contract_address, processing_time))
        results_df = pd.DataFrame(results, columns=['Address', 'Time'])
        # results_df.to_csv('data/output/processing_times_sky.csv', index=False)

    # with open('src/174/adversarial_contracts_bytecode/0xb08ccb39741d746dd1818641900f182448eb5e41.txt', 'r') as file:
    #         bytecode = file.read().strip()
    # process_bytecode(bytecode, output_dir, '0xb08ccb39741d746dd1818641900f182448eb5e41')


if __name__ == "__main__":
    csv_file = 'SOG/src/backup/Incident.csv'
    output_dir = '/home/sandra/DATA/SOG_SET/skyeye/'
    main(csv_file, output_dir)
