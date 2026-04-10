"""RGCN 推理服务。"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from Prototype.backend.utils.file_utils import read_json


class RgcnInferenceError(RuntimeError):
    """RGCN 推理异常。"""


def run_inference(
    python_executable: str,
    script_path: Path,
    graph_json_path: Path,
    model_path: Path,
    output_path: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    """调用独立脚本完成图推理。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        python_executable,
        str(script_path),
        "--graph-json",
        str(graph_json_path),
        "--model-path",
        str(model_path),
        "--output",
        str(output_path),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        raise RgcnInferenceError(completed.stderr.strip() or completed.stdout.strip() or "RGCN 推理失败。")
    if not output_path.exists():
        raise RgcnInferenceError("RGCN 推理完成，但未找到结果文件。")
    return read_json(output_path)
