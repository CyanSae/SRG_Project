"""原型系统配置模块。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOTYPE_ROOT = PROJECT_ROOT / "Prototype"
RUNTIME_ROOT = PROTOTYPE_ROOT / "runtime"
JOBS_ROOT = RUNTIME_ROOT / "jobs"
DOWNLOAD_ROOT = RUNTIME_ROOT / "downloads"
FRONTEND_ROOT = PROTOTYPE_ROOT / "frontend"


@dataclass(frozen=True)
class AppConfig:
    """集中管理原型系统运行时配置。"""

    host: str = os.getenv("PROTOTYPE_HOST", "127.0.0.1")
    port: int = int(os.getenv("PROTOTYPE_PORT", "8000"))
    # 默认内置 Etherscan API Key，必要时仍可通过环境变量覆盖。
    etherscan_api_key: str = os.getenv("ETHERSCAN_API_KEY", "1EYF2RHYIB34DH5SJHPZ2RV1KE7J8WTAU3")
    etherscan_base_url: str = os.getenv("ETHERSCAN_BASE_URL", "https://api.etherscan.io/v2/api")
    sog_python: str = os.getenv("SOG_PYTHON", os.getenv("PROTOTYPE_PYTHON", "python"))
    rgcn_python: str = os.getenv("RGCN_PYTHON", "/home/sandra/anaconda3/envs/rgcn/bin/python")
    model_path: str = os.getenv(
        "RGCN_MODEL_PATH",
        str(PROJECT_ROOT / "RGCN/model/trained_model/hie/best_model_hie_rgcn_2426_393-400.pt"),
    )
    request_timeout_seconds: int = int(os.getenv("PROTOTYPE_REQUEST_TIMEOUT", "300"))
    chain_id: str = os.getenv("ETHERSCAN_CHAIN_ID", "1")


CONFIG = AppConfig()
