import pandas as pd
import os
import json
from sog_builder import SOGBuilder
import dataflow
import tac_cfg
import opcodes
import psutil
import concurrent.futures
import time
import logging

def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024  # Convert to MB

def save_to_json(adjacency_view, file_path):
    edge_type_map = {"data": 0, "control": 1, "effect": 2}
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
    graph_json = {
        "nodes": nodes,
        "edges": edges_list,
        "node_count": len(nodes),
        "edge_count": len(edges_list)
    }
    with open(file_path, 'w') as f:
        json.dump(graph_json, f, indent=4)

def process_with_timeout(data, timeout=30):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(analyse, data)
        try:
            result = future.result(timeout=timeout)
            return False
        except concurrent.futures.TimeoutError:
            print(f"Processing time exceeded the limit of {timeout} seconds for data.")
            return True

def analyse(cfg):
    dataflow.analyse_graph(cfg)

def process_bytecode(bytecode, output_dir, contract_address):
    if bytecode != "" and bytecode != "0x":
        cfg = tac_cfg.TACGraph.from_bytecode(bytecode)
        result = process_with_timeout(cfg)
        if result:
            return
        builder = SOGBuilder(cfg)
        sog = builder.build()
        adjacency_view = sog.adjacency()
        json_file_path = os.path.join(output_dir, f"{contract_address}.json")
        save_to_json(adjacency_view, json_file_path)

def main(parquet_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_parquet(parquet_file)
    results = []
    # logging.info(f"Initial memory usage: {get_memory_usage()} MB")
    for index, (i, row) in enumerate(df.iterrows(), start=0):
        start_time = time.time()
        contract_creation_tx = row['transaction_hash'].hex()
        bytecode = row['init_code'].hex()
        create_index = row['create_index']
        block = row['block_number']
        if bytecode != "" and bytecode != "0x":
            try:
                cfg = tac_cfg.TACGraph.from_bytecode(bytecode)
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(analyse, cfg)
                    try:
                        future.result(timeout=30)
                    except concurrent.futures.TimeoutError:
                        print(f"Processing time exceeded the limit of {30} seconds for the {block}-{create_index} contract {contract_creation_tx}.")
                        continue
                builder = SOGBuilder(cfg)
                sog = builder.build()
                adjacency_view = sog.adjacency()
                json_file_path = os.path.join(output_dir, f"{contract_creation_tx}.json")
                save_to_json(adjacency_view, json_file_path)
                end_time = time.time()
                processing_time = end_time - start_time
                print(f"The {block}-{create_index} contract {contract_creation_tx} done using {processing_time}")
                results.append((contract_creation_tx, processing_time))
                results_df = pd.DataFrame(results, columns=['contract_creation_tx', 'Time'])
                results_df.to_parquet('SOG/data/output/zeroday_processing_times.parquet', index=False)
            except Exception as e:
                logging.error(f"An error occurred while processing the {block}-{create_index} contract {contract_creation_tx}: {e}")
                continue

if __name__ == "__main__":
    parquet_file = '/home/sandra/project/ethereum_contracts__v1_0_0__16000000_to_16799999.parquet'
    output_dir = '/home/sandra/projects/DATA/SOG_SET/zeroday_16'
    main(parquet_file, output_dir)