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
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import unquote, urlparse

from router import get_router_backend_status, run_budget_pipeline
from travel_time import get_today_date
from travel_requirement_parser_llm_agent import _call_llm
from travel_requirement_parser_llm_agent import travel_requirement_parser_llm_agent


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


def _example_departure_date() -> str:
    return (get_today_date() + timedelta(days=7)).isoformat()


def _example_date_range() -> tuple[str, str]:
    start = get_today_date() + timedelta(days=14)
    end = start + timedelta(days=5 - 1)
    return start.isoformat(), end.isoformat()


def _fallback_guidance(result: dict) -> str:
    example_departure_date = _example_departure_date()
    example_start_date, example_end_date = _example_date_range()
    if result.get("stage") == "budget_feasibility":
        trip = result.get("trip_summary") or {}
        budget = (trip.get("budget") or {}).get("total")
        budget_currency = (trip.get("budget") or {}).get("currency") or "CNY"
        duration_days = trip.get("duration_days")
        budget_text = f"{budget} {budget_currency}" if budget else f"your {budget_currency} cap"

        budget_check = result.get("budget_check") or {}
        flight_spend = budget_check.get("selected_total_flight_spend")
        if flight_spend is None:
            flight_spend = ((result.get("rough_budget_check") or {}).get("normal_estimate") or {}).get("transport")
        flight_line = "Flight costs are too high for this budget."
        if isinstance(flight_spend, (int, float)) and float(flight_spend) > 0:
            flight_line = f"Flight costs alone are about {int(round(float(flight_spend)))} {budget_currency} roundtrip."

        suggested_budget_text = f"more than {budget_text}"
        if isinstance(budget, (int, float)) and float(budget) > 0:
            suggested_budget = int(round(float(budget) * 1.2 / 1000.0) * 1000)
            if suggested_budget <= int(round(float(budget))):
                suggested_budget = int(round(float(budget))) + 1000
            suggested_budget_text = f"{suggested_budget} {budget_currency}"

        shorten_example = "a shorter trip"
        if isinstance(duration_days, (int, float)) and int(duration_days) > 1:
            current_days = int(duration_days)
            shorter_days = max(1, current_days - 1)
            day_label = "day" if shorter_days == 1 else "days"
            current_day_label = "day" if current_days == 1 else "days"
            shorten_example = f"{shorter_days} {day_label} instead of {current_days} {current_day_label}"

        return (
            f"This trip is over budget.\n\n"
            f"{flight_line}\n\n"
            "Options:\n"
            "- Increase your budget\n"
            f"- Shorten the trip (for example, {shorten_example})\n"
            "- Remove paid attractions\n"
            "- Change dates\n\n"
            "Reply with one option.\n"
            f'For example: "Increase budget to {suggested_budget_text}" or '
            f'"Keep {budget_text} but remove all paid activities."'
        )

    parsed = result.get("parsed_request") or {}
    missing = (parsed.get("missing_fields") or {}).get("critical") or []
    labels = {
        "origin": "departure city",
        "destination_or_country": "destination city or country",
        "travel_dates_or_duration": "travel dates or trip length",
        "travel_dates.start": "exact departure date",
        "budget.total": "total budget",
    }
    missing_text = ", ".join(labels.get(item, item) for item in missing)
    if missing_text:
        if missing == ["travel_dates.start"] and (parsed.get("duration_days") or (parsed.get("travel_dates") or {}).get("text")):
            return (
                "The departure date is missing. If this trip is starting tomorrow, "
                "please reply with \"starting tomorrow\".\n\n"
                "If not, please provide the exact departure date in YYYY-MM-DD format.\n\n"
                f"For example: {example_departure_date}."
            )
        return (
            f"Please provide {missing_text}.\n\n"
            f"For example: {example_start_date} to {example_end_date}, 2 people, 15000 CNY."
        )
    return (
        "Please provide your departure city, destination, travel dates or trip length, traveler count, and budget."
    )


def _build_guidance_prompt(user_request: str, result: dict) -> str:
    default_example_date = _example_departure_date()
    return f"""
You are a friendly travel planning intake assistant.

The planner could not create a report from the user's request yet. Your job is to help the user revise it into a complete request for the downstream travel planning system.

If you need an example explicit departure date, use {default_example_date}.

Rules:
1. Reply in English.
2. Do not say the request is simply rejected.
3. Do not rewrite or normalize the user's wording, number style, punctuation, capitalization, or formatting.
4. Identify missing or unreasonable information in plain language.
5. Ask focused guiding questions only when required information is missing or unreasonable. Do not ask the user to confirm information that is already present.
6. Do not ask confirmation questions such as "Is this trip starting tomorrow?" or "Do you mean...?" unless the request is genuinely ambiguous and cannot proceed safely.
7. When one exact field is missing, ask directly for that field in one sentence, without restating the whole request unless absolutely necessary.
8. If you give an example departure date, use {default_example_date} by default. Do not give past dates.
9. Do not produce a rewritten full request template unless the user explicitly asks for help rewriting.
10. If the user is continuing a previous request or only added one missing detail, ask only for the next missing detail and keep the reply short.
11. Cover these required downstream information points when relevant:
   - departure city
   - destination city or country
   - travel dates or trip length
   - number of travelers
   - total budget and currency
   - useful preferences such as hotel level/location, food, transport, attractions, pace
12. Keep it concise, under 80 words when only one field is missing, otherwise under 140 words.
13. If the failure stage is "budget_feasibility", treat the user's budget as a hard cap. Say the system already tried cheaper flights, budget hotels, lower-cost meals, and fewer paid activities when applicable. Do not suggest accepting an over-budget plan. Ask the user to either increase the budget, shorten the trip, remove paid activities, or relax destination/date constraints.

User request:
{user_request}

Planner failure result:
{_compact_json(result)}
""".strip()


def _build_guidance(user_request: str, result: dict) -> str:
    parsed = result.get("parsed_request") or {}
    missing = (parsed.get("missing_fields") or {}).get("critical") or []
    if missing:
        return _fallback_guidance(result)
    if result.get("stage") == "budget_feasibility":
        return _fallback_guidance(result)
    try:
        guidance = _call_llm(_build_guidance_prompt(user_request, result)).strip()
        return guidance or _fallback_guidance(result)
    except Exception:
        return _fallback_guidance(result)


def _run_intake_check(user_request: str) -> Dict[str, Any]:
    try:
        parsed_request = travel_requirement_parser_llm_agent(user_request)
        parse_status = str(parsed_request.get("parse_status") or "").lower()
        parse_error = ((parsed_request.get("meta") or {}).get("error") or "").strip()
        if parse_status == "failed" or parse_error:
            return {
                "success": True,
                "ready": True,
                "guidance": None,
                "parsed_request": parsed_request,
                "missing_fields": parsed_request.get("missing_fields", {}),
                "intake_error": parse_error or "Travel intake parsing failed.",
            }
        missing_fields = (parsed_request.get("missing_fields") or {}).get("critical") or []
        is_ready = bool(parsed_request.get("is_ready_for_planning")) and not missing_fields
        result = {
            "success": is_ready,
            "stage": "agent1_parse" if not is_ready else "intake_ready",
            "error": None if is_ready else "The request needs clarification before planning.",
            "parsed_request": parsed_request,
        }
        return {
            "success": True,
            "ready": is_ready,
            "guidance": None if is_ready else _build_guidance(user_request, result),
            "parsed_request": parsed_request,
            "missing_fields": parsed_request.get("missing_fields", {}),
        }
    except Exception as exc:
        result = {
            "success": False,
            "stage": "frontend_validation",
            "error": str(exc),
        }
        return {
            "success": True,
            "ready": True,
            "guidance": None,
            "parsed_request": {},
            "missing_fields": {"critical": [], "optional": []},
            "intake_error": str(exc),
        }


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
            _json_response(self, 200, _run_intake_check(user_request))
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
            failure_status = 422 if response["stage"] in {"agent1_parse", "agent1_validation", "budget_feasibility"} else 500
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


def _frontend_dependencies_ready() -> bool:
    node_modules_dir = FRONTEND_DIR / "node_modules"
    if not node_modules_dir.exists():
        return False

    next_package = node_modules_dir / "next" / "package.json"
    if not next_package.exists():
        return False

    bin_dir = node_modules_dir / ".bin"
    next_bin = bin_dir / ("next.cmd" if os.name == "nt" else "next")
    return next_bin.exists()


def _ensure_frontend_dependencies() -> None:
    if _frontend_dependencies_ready():
        return
    print("Frontend dependencies are missing or incomplete. Installing them now...")
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
            if _wait_for_port("127.0.0.1", FRONTEND_PORT, timeout_s=1.0):
                print(f"TravelAI frontend already running: {frontend_url}")
                webbrowser.open(frontend_url)
            else:
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

        while True:
            if frontend_process is not None and frontend_process.poll() is not None:
                print("Frontend process exited. Keeping API server alive so an existing frontend can still connect.")
                frontend_process = None
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
