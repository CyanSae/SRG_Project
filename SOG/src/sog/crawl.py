import time

import pandas as pd
import requests


def read_csv(file_path):
    df = pd.read_csv(file_path)
    print("read done")
    # return df['contract_address'].tolist()
    return df


def fetch_with_retry(url, params, retries=3, delay=5):
    for i in range(retries):
        try:
            response = requests.get(url, params=params, verify=False)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            print(f"Attempt {i + 1} failed: {e}")
            time.sleep(delay)
    raise Exception("All retry attempts failed")


def fetch_contract_bytecode(address, api_key):
    url = f"https://api.etherscan.io/api"
    params = {
        'module': 'proxy',
        'action': 'eth_getCode',
        'address': address,
        'tag': 'latest',
        'apikey': api_key
    }
    # response = requests.get(url, params=params, verify=False)
    response = fetch_with_retry(url, params)
    data = response.json()

    if response.status_code == 200:
        print(f"Fetch {address}")
        if data.get('result') is not None:
            return data['result']
        else:
            return ''
    else:
        print(f"Error fetching bytecode for {address}: {data['message']}")
        return None

def fetch_from_trans(hash, api_key):
    url = f"https://api.etherscan.io/api"
    params = {
        'module': 'proxy',
        'action': 'eth_getTransactionByHash',
        'txhash': hash,
        'tag': 'latest',
        'apikey': api_key
    }
    # response = requests.get(url, params=params, verify=False)
    response = fetch_with_retry(url, params)
    data = response.json()

    if response.status_code == 200:
        print(f"Fetch {hash}")
        if data.get('result') is not None:
            return data['result'].get('input')
        else:
            return ''
    else:
        print(f"Error fetching bytecode for {hash}: {data['message']}")
        return None

def get_by_trans(csv_file, api_key, output_file):
    df = read_csv(csv_file)
    hashs = df['contract_creation_tx'].tolist()
    labels_dict = df.set_index('contract_creation_tx')['malicious'].to_dict()
    results = []

    for hash in hashs:
        bytecode = fetch_from_trans(hash, api_key)
        if bytecode:
            results.append({'contract_creation_tx': hash, 'malicious': labels_dict.get(hash, None), 'bytecode': bytecode})

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)

def get_by_codes(csv_file, api_key, output_file):
    df = read_csv(csv_file)
    addresses = df['contract_address'].tolist()
    results = []

    for address in addresses:
        bytecode = fetch_contract_bytecode(address, api_key)
        if bytecode:
            results.append({'contract_address': address, 'malicious': 1,'bytecode': bytecode})

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)

def main(csv_file, api_key, output_file):
    # get_by_codes(csv_file, api_key, output_file)
    get_by_trans(csv_file, api_key, output_file)


# Etherscan API密钥
api_key = '1EYF2RHYIB34DH5SJHPZ2RV1KE7J8WTAU3'
csv_file = 'RGCN/shuffled_dataset/creation_1346_shuffled.csv'
output_file = 'RGCN/shuffled_dataset/creation_1346_shuffled_connection.csv'

main(csv_file, api_key, output_file)
