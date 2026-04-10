"""以太坊合约创建字节码抓取服务。"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


ADDRESS_PATTERN = re.compile(r"^0x[a-fA-F0-9]{40}$")
BYTECODE_PATTERN = re.compile(r"^(0x)?[a-fA-F0-9]+$")


@dataclass
class FetchResult:
    """字节码抓取结果。"""

    contract_address: str
    creation_tx: str
    creation_bytecode: str


class FetchError(RuntimeError):
    """字节码抓取异常。"""


def normalize_contract_address(address: str) -> str:
    """标准化并校验合约地址。"""

    value = (address or "").strip()
    if not ADDRESS_PATTERN.fullmatch(value):
        raise FetchError("合约地址格式非法，请输入 0x 开头的 40 位地址。")
    return value


def normalize_bytecode(bytecode: str) -> str:
    """标准化并校验字节码。"""

    value = (bytecode or "").strip()
    if not value:
        raise FetchError("字节码不能为空。")
    if not BYTECODE_PATTERN.fullmatch(value):
        raise FetchError("字节码格式非法，只允许十六进制字符。")
    if not value.startswith("0x"):
        value = f"0x{value}"
    if len(value) <= 2:
        raise FetchError("字节码长度过短，无法进行分析。")
    return value


def _http_get_json(base_url: str, params: dict[str, Any], timeout: int = 30) -> Any:
    """执行 GET 请求并返回 JSON。"""

    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "PrototypeMaliciousContractDetector/1.0",
            "Accept": "application/json",
        },
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read().decode("utf-8")
    return json.loads(payload)


def fetch_creation_bytecode_from_address(
    address: str,
    api_key: str,
    base_url: str,
    chain_id: str,
    timeout: int = 30,
) -> FetchResult:
    """通过 Etherscan API 根据合约地址抓取创建交易与创建字节码。"""

    if not api_key:
        raise FetchError("未配置 ETHERSCAN_API_KEY，无法根据合约地址抓取创建字节码。")

    normalized_address = normalize_contract_address(address)
    creation_payload = _http_get_json(
        base_url,
        {
            "chainid": chain_id,
            "module": "contract",
            "action": "getcontractcreation",
            "contractaddresses": normalized_address,
            "apikey": api_key,
        },
        timeout=timeout,
    )

    result = creation_payload.get("result")
    if not isinstance(result, list) or not result:
        raise FetchError("未能查询到该合约的创建交易信息。")

    creation_info = result[0]
    creation_tx = creation_info.get("txHash") or creation_info.get("txhash")
    if not creation_tx:
        raise FetchError("创建交易哈希缺失，无法继续抓取。")

    tx_payload = _http_get_json(
        base_url,
        {
            "chainid": chain_id,
            "module": "proxy",
            "action": "eth_getTransactionByHash",
            "txhash": creation_tx,
            "apikey": api_key,
        },
        timeout=timeout,
    )

    tx_result = tx_payload.get("result") or {}
    creation_bytecode = tx_result.get("input") or ""
    creation_bytecode = normalize_bytecode(creation_bytecode)
    return FetchResult(
        contract_address=normalized_address,
        creation_tx=creation_tx,
        creation_bytecode=creation_bytecode,
    )
