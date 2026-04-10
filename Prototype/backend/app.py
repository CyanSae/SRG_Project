"""原型系统 HTTP 服务入口。"""

from __future__ import annotations

import json
import mimetypes
import traceback
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from Prototype.backend.config import CONFIG, DOWNLOAD_ROOT, FRONTEND_ROOT, PROJECT_ROOT
from Prototype.backend.services.analyzer_service import analyze_batch, analyze_contract
from Prototype.backend.services.bytecode_fetcher import FetchError


class PrototypeRequestHandler(SimpleHTTPRequestHandler):
    """处理静态资源与 API 请求。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND_ROOT), **kwargs)

    def _send_json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length) if content_length else b"{}"
        if not raw_body:
            return {}
        return json.loads(raw_body.decode("utf-8"))

    def _serve_download(self, target: Path) -> None:
        if not target.exists():
            self.send_error(HTTPStatus.NOT_FOUND, "文件不存在")
            return

        content = target.read_bytes()
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Content-Disposition", f'attachment; filename="{target.name}"')
        self.end_headers()
        self.wfile.write(content)

    def _serve_file_inline(self, target: Path, content_type: str) -> None:
        """以内联方式返回文件内容。"""

        if not target.exists():
            self.send_error(HTTPStatus.NOT_FOUND, "文件不存在")
            return
        content = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            self._send_json(
                {
                    "status": "ok",
                    "service": "Prototype Malicious Contract Detector",
                    "modelPath": CONFIG.model_path,
                    "projectRoot": str(PROJECT_ROOT),
                }
            )
            return

        if parsed.path.startswith("/api/jobs/") and parsed.path.endswith("/srg"):
            job_id = parsed.path.split("/")[3]
            self._serve_download(DOWNLOAD_ROOT / f"{job_id}.json")
            return

        if parsed.path.startswith("/api/jobs/") and parsed.path.endswith("/srg-json"):
            job_id = parsed.path.split("/")[3]
            self._serve_file_inline(DOWNLOAD_ROOT / f"{job_id}.json", "application/json; charset=utf-8")
            return

        if parsed.path == "/shared-lib/vis-network.min.js":
            self._serve_file_inline(
                PROJECT_ROOT / "lib/vis-9.1.2/vis-network.min.js",
                "application/javascript; charset=utf-8",
            )
            return

        if parsed.path == "/shared-lib/vis-network.css":
            self._serve_file_inline(
                PROJECT_ROOT / "lib/vis-9.1.2/vis-network.css",
                "text/css; charset=utf-8",
            )
            return

        if parsed.path in {"/", "/index.html"}:
            self.path = "/index.html"
        elif parsed.path == "/graph.html":
            self.path = "/graph.html"
        return super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path not in {"/api/analyze", "/api/analyze/batch"}:
            self.send_error(HTTPStatus.NOT_FOUND, "接口不存在")
            return

        try:
            payload = self._read_json_body()
            if parsed.path == "/api/analyze/batch":
                result = analyze_batch(payload)
            else:
                result = analyze_contract(payload)
            self._send_json({"success": True, "data": result})
        except (ValueError, FetchError) as error:
            self._send_json({"success": False, "error": str(error)}, status=HTTPStatus.BAD_REQUEST)
        except Exception as error:  # noqa: BLE001
            self._send_json(
                {
                    "success": False,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )


def run_server() -> None:
    """启动 HTTP 服务。"""

    server = ThreadingHTTPServer((CONFIG.host, CONFIG.port), PrototypeRequestHandler)
    print(f"Prototype 服务已启动: http://{CONFIG.host}:{CONFIG.port}")
    server.serve_forever()


if __name__ == "__main__":
    run_server()
