"""文件与路径工具函数。"""

from __future__ import annotations

import json
import secrets
from pathlib import Path
from typing import Any


def ensure_dir(path: Path) -> Path:
    """确保目录存在。"""

    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> Any:
    """读取 JSON 文件。"""

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, data: Any) -> None:
    """写入 JSON 文件。"""

    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def make_job_id() -> str:
    """生成简短且足够随机的任务编号。"""

    return secrets.token_hex(8)
