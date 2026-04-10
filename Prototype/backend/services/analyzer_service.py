"""检测总流程服务。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from Prototype.backend.config import CONFIG, DOWNLOAD_ROOT, JOBS_ROOT, PROJECT_ROOT
from Prototype.backend.services.bytecode_fetcher import (
    FetchError,
    fetch_creation_bytecode_from_address,
    normalize_bytecode,
)
from Prototype.backend.services.rgcn_service import run_inference
from Prototype.backend.services.sog_service import build_srg
from Prototype.backend.utils.file_utils import ensure_dir, make_job_id, write_json


GRANULARITY_LABELS = {
    "binary": "是否恶意",
    "category": "恶意大类",
    "subtype": "具体类型",
}


def _normalize_granularity(payload: dict[str, Any]) -> str:
    """校验检测粒度。"""

    granularity = payload.get("granularity", "subtype")
    if granularity not in GRANULARITY_LABELS:
        raise ValueError("检测粒度非法。")
    return granularity


def _normalize_source_type(payload: dict[str, Any]) -> str:
    """校验输入类型。"""

    source_type = payload.get("sourceType", "bytecode")
    if source_type not in {"address", "bytecode"}:
        raise ValueError("输入类型非法。")
    return source_type


def _build_display_result(prediction: dict[str, Any], granularity: str) -> dict[str, Any]:
    """根据用户选择的粒度组装展示结果。"""

    if granularity == "binary":
        return {
            "label": prediction["binary"]["label_zh"],
            "confidence": prediction["binary"]["confidence"],
            "message": prediction["binary"]["summary"],
        }
    if granularity == "category":
        return {
            "label": prediction["category"]["label_zh"],
            "confidence": prediction["category"]["confidence"],
            "message": prediction["category"]["summary"],
        }
    return {
        "label": prediction["subtype"]["label_zh"],
        "confidence": prediction["subtype"]["confidence"],
        "message": prediction["subtype"]["summary"],
    }


def analyze_contract(payload: dict[str, Any]) -> dict[str, Any]:
    """执行完整检测流程。"""

    source_type = _normalize_source_type(payload)
    granularity = _normalize_granularity(payload)

    job_id = make_job_id()
    job_dir = ensure_dir(JOBS_ROOT / job_id)
    srg_dir = ensure_dir(job_dir / "srg")
    result_dir = ensure_dir(job_dir / "result")

    identifier = f"contract_{job_id}"
    contract_address = None
    creation_tx = None
    source_desc = "用户上传字节码"

    if source_type == "address":
        fetch_result = fetch_creation_bytecode_from_address(
            address=payload.get("address", ""),
            api_key=CONFIG.etherscan_api_key,
            base_url=CONFIG.etherscan_base_url,
            chain_id=CONFIG.chain_id,
            timeout=30,
        )
        contract_address = fetch_result.contract_address
        creation_tx = fetch_result.creation_tx
        bytecode = fetch_result.creation_bytecode
        source_desc = "根据合约地址抓取创建字节码"
    else:
        bytecode = normalize_bytecode(payload.get("bytecode", ""))

    srg_json_path = build_srg(
        python_executable=CONFIG.sog_python,
        script_path=PROJECT_ROOT / "Prototype/scripts/build_srg.py",
        bytecode=bytecode,
        output_dir=srg_dir,
        identifier=identifier,
        timeout_seconds=CONFIG.request_timeout_seconds,
    )

    inference_output_path = result_dir / "prediction.json"
    prediction = run_inference(
        python_executable=CONFIG.rgcn_python,
        script_path=PROJECT_ROOT / "Prototype/scripts/run_inference.py",
        graph_json_path=srg_json_path,
        model_path=Path(CONFIG.model_path),
        output_path=inference_output_path,
        timeout_seconds=CONFIG.request_timeout_seconds,
    )

    download_target = DOWNLOAD_ROOT / f"{job_id}.json"
    download_target.write_text(srg_json_path.read_text(encoding="utf-8"), encoding="utf-8")

    response = {
        "jobId": job_id,
        "createdAt": datetime.utcnow().isoformat() + "Z",
        "granularity": granularity,
        "granularityLabel": GRANULARITY_LABELS[granularity],
        "input": {
            "sourceType": source_type,
            "sourceDescription": source_desc,
            "contractAddress": contract_address,
            "creationTx": creation_tx,
            "bytecodeLength": len(bytecode),
        },
        "prediction": prediction,
        "displayResult": _build_display_result(prediction, granularity),
        "downloads": {
            "srgJson": f"/api/jobs/{job_id}/srg",
            "srgGraphData": f"/api/jobs/{job_id}/srg-json",
            "srgVisualizer": f"/graph.html?jobId={job_id}",
        },
        "paths": {
            "jobDir": str(job_dir.relative_to(PROJECT_ROOT)),
            "srgJson": str(srg_json_path.relative_to(PROJECT_ROOT)),
            "predictionJson": str(inference_output_path.relative_to(PROJECT_ROOT)),
        },
    }
    write_json(job_dir / "response.json", response)
    return response


def analyze_batch(payload: dict[str, Any]) -> dict[str, Any]:
    """执行批量检测。"""

    source_type = _normalize_source_type(payload)
    granularity = _normalize_granularity(payload)
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("批量检测至少需要 1 条输入数据。")

    batch_job_id = make_job_id()
    batch_dir = ensure_dir(JOBS_ROOT / f"batch_{batch_job_id}")
    results: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            failed.append(
                {
                    "index": index,
                    "error": "批量输入项格式非法。",
                }
            )
            continue

        sub_payload = {
            "sourceType": source_type,
            "granularity": granularity,
            "address": item.get("address", ""),
            "bytecode": item.get("bytecode", ""),
        }
        try:
            result = analyze_contract(sub_payload)
            results.append(
                {
                    "index": index,
                    "jobId": result["jobId"],
                    "displayResult": result["displayResult"],
                    "input": result["input"],
                    "downloads": result["downloads"],
                    "prediction": result["prediction"],
                }
            )
        except Exception as error:  # noqa: BLE001
            failed.append(
                {
                    "index": index,
                    "error": str(error),
                    "rawInput": item,
                }
            )

    summary = {
        "total": len(items),
        "success": len(results),
        "failed": len(failed),
        "malicious": sum(1 for item in results if item["prediction"]["binary"]["code"] == "malicious"),
        "benign": sum(1 for item in results if item["prediction"]["binary"]["code"] == "benign"),
    }
    response = {
        "batchJobId": batch_job_id,
        "createdAt": datetime.utcnow().isoformat() + "Z",
        "sourceType": source_type,
        "granularity": granularity,
        "granularityLabel": GRANULARITY_LABELS[granularity],
        "summary": summary,
        "results": results,
        "failedItems": failed,
    }
    write_json(batch_dir / "response.json", response)
    return response
