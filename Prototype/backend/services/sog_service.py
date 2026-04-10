"""SRG 生成服务。"""

from __future__ import annotations

import subprocess
from pathlib import Path


class SogBuildError(RuntimeError):
    """SRG 生成异常。"""


def build_srg(
    python_executable: str,
    script_path: Path,
    bytecode: str,
    output_dir: Path,
    identifier: str,
    timeout_seconds: int,
) -> Path:
    """调用独立脚本生成 SRG JSON。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        python_executable,
        str(script_path),
        "--bytecode",
        bytecode,
        "--output-dir",
        str(output_dir),
        "--identifier",
        identifier,
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        raise SogBuildError(completed.stderr.strip() or completed.stdout.strip() or "SRG 生成失败。")

    json_path = output_dir / f"{identifier}.json"
    if not json_path.exists():
        raise SogBuildError("SRG 生成脚本执行成功，但未找到输出的 JSON 文件。")
    return json_path
