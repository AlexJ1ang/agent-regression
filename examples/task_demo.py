"""Local deterministic test double, not a model or a production agent.

python examples/task_demo.py --port 8765
python examples/task_demo.py --port 8765 --fault wrong-device

Faults are deliberately injected to verify the test framework's behavior.
All data is fictional; each new conversation has its own mutable workspace.
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import unquote

ENERGY = {
    "A": {"2026-08": 100, "2026-09": 150},
    "B": {"2026-08": 200, "2026-09": 300},
}


class DemoService:
    def __init__(self, fault: str | None = None) -> None:
        self.fault = fault
        self.sessions: dict[str, dict[str, Any]] = {}
        self.lock = threading.RLock()

    def _query_energy(self, session: dict[str, Any], device_id: str, month: str) -> dict:
        return {"device_id": device_id, "month": month,
                "liters": ENERGY[device_id][month]}

    def _create_work_order(
        self, session: dict[str, Any], device_id: str, priority: str,
    ) -> dict:
        order = {"id": f"W-{len(session['work_orders']) + 1}", "device_id": device_id,
                 "priority": priority, "status": "open"}
        session["work_orders"].append(order)
        return order

    def _get_work_order(self, session: dict[str, Any], order_id: str) -> dict:
        return next(order for order in session["work_orders"] if order["id"] == order_id)

    def chat(self, request: dict[str, Any]) -> dict[str, Any]:
        message = request.get("query")
        if not isinstance(message, str):
            raise ValueError("query must be a string")
        with self.lock:
            conversation_id = request.get("conversation_id")
            if conversation_id is None:
                conversation_id = uuid.uuid4().hex
                self.sessions[conversation_id] = {"device_id": None, "month": None,
                                                  "work_orders": [], "pending_order": False}
            if conversation_id not in self.sessions:
                raise ValueError("unknown conversation")
            session = self.sessions[conversation_id]
            calls: list[dict[str, Any]] = []

            def call(name: str, **arguments: Any) -> dict:
                function = getattr(self, f"_{name}")
                result = function(session, **arguments)
                calls.append({"call_id": f"call-{len(calls) + 1}",
                              "name": name, "arguments": arguments})
                return result

            device = re.search(r"\b([AB])\b", message)
            month = re.search(r"2026-(?:08|09)", message)
            if device:
                session["device_id"] = device.group(1)
            if month:
                session["month"] = month.group(0)

            if "创建" in message and "工单" in message and "确认" not in message:
                session["pending_order"] = True
                answer = {"question": "请提供优先级，再确认创建工单"}
            elif session["pending_order"] and "确认" in message:
                if not session["device_id"] or "high" not in message:
                    answer = {"question": "需要设备和优先级 high"}
                else:
                    answer = call("create_work_order", device_id=session["device_id"], priority="high")
                    session["pending_order"] = False
            elif "工单" in message and session["work_orders"]:
                answer = call("get_work_order", order_id=session["work_orders"][-1]["id"])
            elif session["device_id"] and session["month"]:
                device_id = "A" if self.fault == "wrong-device" else session["device_id"]
                query_month = "2026-08" if self.fault == "wrong-month" else session["month"]
                answer = call("query_energy", device_id=device_id, month=query_month)
            else:
                answer = {"question": "请提供设备 A/B 和月份 2026-08/2026-09"}

            response = {"answer": json.dumps(answer, ensure_ascii=False),
                        "conversation_id": conversation_id}
            if self.fault != "no-tools":
                response["tool_calls"] = calls
            return response

    def state(self, conversation_id: str) -> dict[str, Any]:
        with self.lock:
            if conversation_id not in self.sessions:
                raise ValueError("unknown conversation")
            # A snapshot of the test environment, independently of answer generation.
            return json.loads(json.dumps(self.sessions[conversation_id]))


def make_server(port: int = 8765, fault: str | None = None) -> ThreadingHTTPServer:
    service = DemoService(fault)

    class Handler(BaseHTTPRequestHandler):
        def respond(self, status: int, payload: dict[str, Any]) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self) -> None:
            if self.path != "/chat":
                self.respond(404, {"error": "not found"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 65536:
                    raise ValueError("invalid request size")
                payload = json.loads(self.rfile.read(size))
                if not isinstance(payload, dict):
                    raise ValueError("request must be an object")
                self.respond(200, service.chat(payload))
            except (ValueError, TypeError, KeyError):
                self.respond(400, {"error": "invalid request or conversation"})

        def do_GET(self) -> None:
            if not self.path.startswith("/state/"):
                self.respond(404, {"error": "not found"})
                return
            try:
                self.respond(200, service.state(unquote(self.path[len("/state/"):])) )
            except ValueError:
                self.respond(404, {"error": "unknown conversation"})

        def log_message(self, format: str, *args: Any) -> None:
            pass

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--fault", choices=["wrong-device", "wrong-month", "no-tools"])
    args = parser.parse_args()
    server = make_server(args.port, args.fault)
    print(f"Local test double: http://127.0.0.1:{server.server_port} (fault={args.fault})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
