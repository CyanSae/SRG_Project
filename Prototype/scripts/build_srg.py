"""独立的 SRG 生成脚本。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "Prototype/runtime/mplconfig"))
SOG_SRC_ROOT = PROJECT_ROOT / "SOG/src"
SOG_GRAPH_ROOT = PROJECT_ROOT / "SOG/src/sog"

sys.path.insert(0, str(SOG_SRC_ROOT))
sys.path.insert(0, str(SOG_GRAPH_ROOT))

from data_process_connection import process_bytecode  # noqa: E402


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="根据创建字节码生成 SRG JSON。")
    parser.add_argument("--bytecode", required=True, help="待分析的创建字节码。")
    parser.add_argument("--output-dir", required=True, help="SRG 输出目录。")
    parser.add_argument("--identifier", required=True, help="输出文件标识。")
    return parser.parse_args()


def main() -> int:
    """脚本主入口。"""

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    process_bytecode(args.bytecode.strip(), str(output_dir), args.identifier)
    output_file = output_dir / f"{args.identifier}.json"
    if not output_file.exists():
        raise RuntimeError("SRG 生成失败，未得到输出文件。")
    print(output_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
