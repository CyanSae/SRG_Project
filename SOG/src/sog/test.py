import requests

url = "https://api.etherscan.io/api"
params = {
    "module": "proxy",
    "action": "eth_blockNumber",
    "apikey": '1EYF2RHYIB34DH5SJHPZ2RV1KE7J8WTAU3'  # 用 Etherscan 提供的演示 API key
}

proxies = {
    "http": "http://192.168.50.119:7890",
    "https": "http://192.168.50.119:7890"
}

try:
    print("发送请求中...")
    r = requests.get(url, params=params, proxies=proxies, timeout=30)
    print("✅ 成功！响应：", r.json())
except requests.exceptions.RequestException as e:
    print("❌ 请求失败：", e)

