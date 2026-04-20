from __future__ import annotations

import json
import mimetypes
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import unquote, urlparse

from router import get_router_backend_status, run_budget_pipeline
from travel_requirement_parser_llm_agent import _call_llm


BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "front-end"
REPORTS_DIR = BASE_DIR / "travel_reports"
API_HOST = os.getenv("TRAVEL_API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("TRAVEL_API_PORT", "8000"))
FRONTEND_PORT = int(os.getenv("TRAVEL_FRONTEND_PORT", "3000"))


def _compact_json(value: Any, limit: int = 2500) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return text
    return text[:limit] + "...(truncated)"


def _fallback_guidance(result: dict) -> str:
    parsed = result.get("parsed_request") or {}
    missing = (parsed.get("missing_fields") or {}).get("critical") or []
    labels = {
        "origin": "departure city",
        "destination_or_country": "destination city or country",
        "travel_dates_or_duration": "travel dates or trip length",
        "budget.total": "total budget",
    }
    missing_text = ", ".join(labels.get(item, item) for item in missing)
    understood = [
        f"Departure city: {parsed.get('origin') or 'not provided'}",
        f"Destination: {', '.join(parsed.get('destination') or []) if parsed.get('destination') else parsed.get('country') or 'not provided'}",
        f"Dates or duration: {(parsed.get('travel_dates') or {}).get('text') or (parsed.get('travel_dates') or {}).get('start') or parsed.get('duration_days') or 'not provided'}",
        f"Travelers: {parsed.get('num_people') or 1}",
        f"Budget: {(parsed.get('budget') or {}).get('total') or 'not provided'} {(parsed.get('budget') or {}).get('currency') or 'CNY'}",
    ]
    if missing_text:
        return (
            "I can help turn this into a complete travel request.\n\n"
            "What I have so far:\n"
            + "\n".join(f"- {item}" for item in understood)
            + "\n\n"
            f"What I still need: {missing_text}.\n\n"
            "Please reply with the missing details, for example:\n"
            "I want to travel from Shanghai to Paris from 2026-06-10 to 2026-06-15 "
            "for 2 people with a budget of 15000 CNY. I prefer clean central hotels and local food."
        )
    return (
        "I can help refine this request before creating the report.\n\n"
        "Please make sure it includes these information points: departure city, destination, travel dates or trip length, "
        "traveler count, total budget and currency, plus hotel, food, transport, or attraction preferences."
    )


def _build_guidance_prompt(user_request: str, result: dict) -> str:
    return f"""
You are a friendly travel planning intake assistant.

The planner could not create a report from the user's request yet. Your job is to help the user revise it into a complete request for the downstream travel planning system.

Rules:
1. Reply in English.
2. Do not say the request is simply rejected.
3. Summarize what you already understood from the request.
4. Identify missing or unreasonable information in plain language.
5. Ask focused guiding questions for the smallest set of details needed to continue.
6. If this is the first assistant clarification, you may end with one rewritten complete request template tailored to the user's intent.
7. If the user is continuing a previous request or only added one missing detail, do not restart with a full template. Acknowledge the new detail, state what is now locked in, and ask the next missing details as clear fill-in prompts.
8. Make follow-up guidance feel progressive and helpful, for example:
   - "Great, I have Guangzhou -> Paris now."
   - "Next I just need the dates or trip length, total budget, and traveler count."
   - "You can reply in one line, like: June 10 to June 15, 2 people, 15000 CNY."
9. Cover these required downstream information points when relevant:
   - departure city
   - destination city or country
   - travel dates or trip length
   - number of travelers
   - total budget and currency
   - useful preferences such as hotel level/location, food, transport, attractions, pace
10. Keep it concise, under 140 words.

User request:
{user_request}

Planner failure result:
{_compact_json(result)}
""".strip()


def _build_guidance(user_request: str, result: dict) -> str:
    try:
        guidance = _call_llm(_build_guidance_prompt(user_request, result)).strip()
        return guidance or _fallback_guidance(result)
    except Exception:
        return _fallback_guidance(result)


def _fallback_frontend_html() -> bytes:
    return b"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TravelAI Planner</title>
  <style>
    :root { color-scheme: light; --ink:#18201f; --muted:#65716f; --line:#d9e0dd; --paper:#f7f8f4; --panel:#fff; --accent:#007c89; }
    * { box-sizing: border-box; }
    body { margin:0; font-family: Arial, Helvetica, sans-serif; background:var(--paper); color:var(--ink); }
    main { min-height:100vh; display:flex; align-items:center; justify-content:center; padding:24px; }
    .shell { width:min(900px, 100%); }
    h1 { margin:0 0 8px; font-size:32px; }
    p { color:var(--muted); line-height:1.5; }
    form { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:16px; margin-top:18px; }
    textarea { width:100%; min-height:140px; resize:vertical; border:1px solid var(--line); border-radius:8px; padding:12px; font:inherit; }
    button { margin-top:12px; border:0; border-radius:8px; background:var(--accent); color:#fff; padding:11px 16px; font-weight:700; cursor:pointer; }
    button:disabled { opacity:.6; cursor:not-allowed; }
    .result { margin-top:16px; background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:16px; white-space:pre-wrap; }
    a { color:var(--accent); font-weight:700; }
  </style>
</head>
<body>
  <main>
    <div class="shell">
      <h1>TravelAI Planner</h1>
      <p>This fallback page is served by Python because Node.js/npm is not available. It still calls the same router pipeline and opens the generated report.</p>
      <form id="form">
        <textarea id="message" placeholder="Describe your trip, for example: I want to travel from Beijing to Chengdu from 6.19 to 6.21 with a budget of 6000 CNY. I love local food."></textarea>
        <br>
        <button id="submit" type="submit">Generate plan</button>
      </form>
      <div id="result" class="result" style="display:none;"></div>
    </div>
  </main>
  <script>
    const form = document.getElementById('form');
    const message = document.getElementById('message');
    const submit = document.getElementById('submit');
    const result = document.getElementById('result');
    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      const text = message.value.trim();
      if (!text) return;
      submit.disabled = true;
      result.style.display = 'block';
      result.textContent = 'Running router pipeline and generating report...';
      try {
        const response = await fetch('/api/plan', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: text })
        });
        const data = await response.json();
        if (!response.ok || !data.success) throw new Error(data.error || 'Planning failed.');
        const trip = data.trip_summary || {};
        const destination = Array.isArray(trip.destination) ? trip.destination.join(', ') : '';
        result.innerHTML = [
          '<strong>Plan generated.</strong>',
          '',
          'Route: ' + (trip.origin || 'Origin') + ' -> ' + (destination || 'destination'),
          'Days: ' + (trip.duration_days || 'unknown'),
          data.report_url ? '<a href="' + data.report_url + '" target="_blank" rel="noreferrer">Open report</a>' : 'No report URL returned.'
        ].join('<br>');
      } catch (error) {
        result.textContent = 'The plan could not be generated.\\n\\n' + (error && error.message ? error.message : String(error));
      } finally {
        submit.disabled = false;
      }
    });
  </script>
</body>
</html>"""


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: Dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")
    handler.end_headers()
    handler.wfile.write(body)


def _report_url_from_result(result: Dict[str, Any]) -> Optional[str]:
    report = result.get("report") or {}
    if not isinstance(report, dict) or not report.get("success"):
        return None

    path_text = report.get("path") or report.get("file")
    if not path_text:
        return None

    path = Path(path_text)
    if not path.is_absolute():
        path = BASE_DIR / path

    try:
        relative = path.resolve().relative_to(REPORTS_DIR.resolve())
    except Exception:
        return None

    return f"http://{API_HOST}:{API_PORT}/reports/{relative.as_posix()}"


class TravelApiHandler(BaseHTTPRequestHandler):
    server_version = "TravelAI/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stdout.write("[api] " + fmt % args + "\n")

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"", "/"}:
            data = _fallback_frontend_html()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if parsed.path == "/api/status":
            _json_response(self, 200, {"success": True, "status": get_router_backend_status()})
            return

        if parsed.path.startswith("/reports/"):
            self._serve_report(parsed.path[len("/reports/"):])
            return

        _json_response(self, 404, {"success": False, "error": "Unknown endpoint."})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path not in {"/api/plan", "/api/guidance"}:
            _json_response(self, 404, {"success": False, "error": "Unknown endpoint."})
            return

        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            _json_response(self, 400, {"success": False, "error": "Invalid JSON body."})
            return

        user_request = str(payload.get("message") or payload.get("request") or "").strip()
        if not user_request:
            _json_response(self, 400, {"success": False, "error": "Message is required."})
            return

        if parsed.path == "/api/guidance":
            reason = str(payload.get("reason") or "The request needs clarification.").strip()
            guidance_result = {
                "success": False,
                "stage": "frontend_validation",
                "error": reason,
                "parsed_request": {},
            }
            _json_response(
                self,
                200,
                {
                    "success": True,
                    "guidance": _build_guidance(user_request, guidance_result),
                },
            )
            return

        try:
            result = run_budget_pipeline(user_request, auto_open_report=False)
            report_url = _report_url_from_result(result)
            success = bool(result.get("success"))
            response = {
                "success": success,
                "stage": result.get("stage"),
                "error": result.get("error"),
                "guidance": None if success else _build_guidance(user_request, result),
                "warnings": result.get("warnings", []),
                "trip_summary": result.get("trip_summary", {}),
                "itinerary_summary": (result.get("itinerary") or {}).get("itinerary_summary", {}),
                "report": result.get("report", {}),
                "report_url": report_url,
            }
            failure_status = 422 if response["stage"] in {"agent1_parse", "agent1_validation"} else 500
            _json_response(self, 200 if success else failure_status, response)
        except Exception as exc:
            _json_response(self, 500, {"success": False, "error": str(exc)})

    def _serve_report(self, report_path: str) -> None:
        safe_name = unquote(report_path).replace("\\", "/").lstrip("/")
        target = (REPORTS_DIR / safe_name).resolve()
        try:
            target.relative_to(REPORTS_DIR.resolve())
        except Exception:
            _json_response(self, 403, {"success": False, "error": "Forbidden report path."})
            return

        if not target.is_file():
            _json_response(self, 404, {"success": False, "error": "Report not found."})
            return

        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)


def _start_api_server() -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((API_HOST, API_PORT), TravelApiHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _node_command(name: str) -> str | None:
    command = shutil.which(name)
    if command:
        return command
    if os.name == "nt":
        return shutil.which(f"{name}.cmd")
    return None


def _frontend_command() -> list[str]:
    pnpm = _node_command("pnpm")
    if pnpm:
        return [pnpm, "dev", "--hostname", "127.0.0.1", "--port", str(FRONTEND_PORT)]
    npm = _node_command("npm")
    if npm:
        return [npm, "run", "dev", "--", "--hostname", "127.0.0.1", "--port", str(FRONTEND_PORT)]
    raise RuntimeError("Neither pnpm nor npm was found. Please install Node.js and pnpm/npm.")


def _install_command() -> list[str]:
    pnpm = _node_command("pnpm")
    if pnpm:
        return [pnpm, "install"]
    npm = _node_command("npm")
    if npm:
        return [npm, "install"]
    raise RuntimeError("Neither pnpm nor npm was found. Please install Node.js and pnpm/npm.")


def _ensure_frontend_dependencies() -> None:
    if (FRONTEND_DIR / "node_modules").exists():
        return
    print("Frontend dependencies are missing. Installing them now...")
    subprocess.check_call(_install_command(), cwd=str(FRONTEND_DIR))


def _wait_for_port(host: str, port: int, timeout_s: float = 45.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            if sock.connect_ex((host, port)) == 0:
                return True
        time.sleep(0.5)
    return False


def _start_frontend() -> subprocess.Popen[Any]:
    if not FRONTEND_DIR.exists():
        raise RuntimeError(f"Frontend folder not found: {FRONTEND_DIR}")

    _ensure_frontend_dependencies()
    command = _frontend_command()
    env = os.environ.copy()
    env["NEXT_PUBLIC_TRAVEL_API_BASE"] = f"http://{API_HOST}:{API_PORT}"
    return subprocess.Popen(command, cwd=str(FRONTEND_DIR), env=env)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    api_server = _start_api_server()
    frontend_process: Optional[subprocess.Popen[Any]] = None
    frontend_url = f"http://127.0.0.1:{FRONTEND_PORT}"
    fallback_url = f"http://{API_HOST}:{API_PORT}/"

    try:
        print(f"TravelAI API: http://{API_HOST}:{API_PORT}")
        try:
            frontend_process = _start_frontend()
            print(f"TravelAI frontend: {frontend_url}")

            if _wait_for_port("127.0.0.1", FRONTEND_PORT):
                webbrowser.open(frontend_url)
            else:
                print("Frontend did not become ready in time. Open this URL after it finishes compiling:")
                print(frontend_url)
        except RuntimeError as exc:
            print(f"Next frontend unavailable: {exc}")
            print(f"Opening Python fallback frontend: {fallback_url}")
            webbrowser.open(fallback_url)

        while frontend_process is None or frontend_process.poll() is None:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping TravelAI...")
    finally:
        api_server.shutdown()
        if frontend_process is not None:
            frontend_process.terminate()
            try:
                frontend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                frontend_process.kill()


if __name__ == "__main__":
    main()
