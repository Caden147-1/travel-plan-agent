# Travel Plan Agent

Travel Plan Agent is an end-to-end AI travel planning system that converts natural-language travel requests into budget-aware, day-by-day itineraries and locally generated HTML reports.

The project combines a Python planning pipeline with a Next.js chat interface. It supports request parsing, destination and route planning, budget allocation, real-time candidate selection for flights, hotels, restaurants, and attractions, itinerary scheduling, report generation, and follow-up refinement when users ask to upgrade parts of an existing plan.

## Features

- Natural-language trip intake with validation for origin, destinations, dates, duration, budget, and preferences.
- Multi-city route planning with explicit city order and duration handling.
- Budget planning across transport, hotels, food, attractions, and buffer.
- Candidate selection for flights, hotels, restaurants, and points of interest.
- Day-by-day itinerary generation with transfers, meals, check-in/check-out events, and timeline overlap correction.
- HTML report generation under `travel_reports/`.
- Chat-based refinement flow for upgrading food, attractions, flights, or hotels based on the latest successful report.
- Refinement state persistence in browser history/localStorage, including budget ceilings and previous plan baselines.
- Frontend chat UI built with Next.js, React, Tailwind CSS, and Radix UI components.

## Architecture

```text
User request
  -> Frontend chat interface
  -> Python API server
  -> Agent 1: requirement parsing and validation
  -> Destination planning
  -> Agent 2: budget allocation
  -> Agent 3: candidate search and selection
  -> Agent 4: itinerary scheduling
  -> HTML report generator
```

Key backend modules:

- `main.py`: local API server, frontend launcher, report serving, and API request handling.
- `router.py`: orchestration layer for the full planning pipeline and refinement context.
- `travel_requirement_parser_llm_agent.py`: natural-language request parsing.
- `destination.py`: destination and city route planning.
- `travel_budget_agent.py`: budget allocation and feasibility checks.
- `agent3_real.py`: real candidate search and selection for flights, hotels, restaurants, and POIs.
- `itinerary_agent.py`: day-by-day itinerary generation and timeline normalization.
- `travel_report_generator.py`: local HTML report generation.
- `front-end/`: Next.js frontend application.

## Requirements

- Python 3.10 or newer.
- Node.js with either `pnpm` or `npm`.
- API keys for live planning:
  - `GOOGLE_MAPS_API_KEY`
  - `SEARCHAPI_API_KEY`
  - `DASHSCOPE_API_KEY` or `QWEN_API_KEY`
- Optional:
  - `HEIGIT_API_KEY` for OpenRouteService-based routing.

Python dependencies are intentionally lightweight. The backend uses the Python standard library plus `requests`.

```bash
pip install requests
```

Frontend dependencies are managed inside `front-end/package.json`.

## Environment Variables

| Variable | Required | Description |
| --- | --- | --- |
| `GOOGLE_MAPS_API_KEY` | Yes | Google Maps and Places data used by candidate search and routing helpers. |
| `SEARCHAPI_API_KEY` | Yes | SearchAPI key used for flight and travel search. |
| `DASHSCOPE_API_KEY` or `QWEN_API_KEY` | Yes | LLM provider key for parsing, budget reasoning, and optional local recommendations. |
| `TRAVEL_LLM_MODEL` | No | LLM model name. Defaults to `qwen-plus`. |
| `TRAVEL_API_HOST` | No | Python API host. Defaults to `127.0.0.1`. |
| `TRAVEL_API_PORT` | No | Python API port. Defaults to `8000`. |
| `TRAVEL_FRONTEND_PORT` | No | Next.js frontend port. Defaults to `3000`. |
| `NEXT_PUBLIC_TRAVEL_API_BASE` | No | Frontend API base URL. Automatically set by `main.py`. |
| `HEIGIT_API_KEY` | No | Optional OpenRouteService key for route duration and distance calls. |
| `TRAVEL_CITY_ORDER_USE_MAP_API` | No | Set to `false` to disable map API ordering for local fallback runs. |

Example PowerShell setup:

```powershell
$env:GOOGLE_MAPS_API_KEY="your-google-key"
$env:SEARCHAPI_API_KEY="your-searchapi-key"
$env:DASHSCOPE_API_KEY="your-dashscope-key"
```

Example macOS/Linux setup:

```bash
export GOOGLE_MAPS_API_KEY="your-google-key"
export SEARCHAPI_API_KEY="your-searchapi-key"
export DASHSCOPE_API_KEY="your-dashscope-key"
```

## Quick Start

From the repository root:

```bash
python main.py
```

This starts the Python API server and launches the Next.js frontend. If frontend dependencies are missing, `main.py` attempts to install them automatically with `pnpm install` or `npm install`.

Default local URLs:

- Frontend: `http://127.0.0.1:3000`
- API: `http://127.0.0.1:8000`

Generated reports are written to:

```text
travel_reports/
```

## Manual Frontend Setup

If you prefer to run the frontend manually:

```bash
cd front-end
npm install
npm run dev -- --hostname 127.0.0.1 --port 3000
```

or, with pnpm:

```bash
cd front-end
pnpm install
pnpm dev --hostname 127.0.0.1 --port 3000
```

When running manually, make sure the frontend can reach the API:

```bash
NEXT_PUBLIC_TRAVEL_API_BASE=http://127.0.0.1:8000
```

## API Endpoints

The local API server exposes:

- `GET /`: lightweight fallback web interface.
- `GET /api/status`: backend capability and API-key status.
- `POST /api/guidance`: request intake validation and missing-field guidance.
- `POST /api/plan`: full planning pipeline.
- `GET /reports/<filename>`: serves generated HTML reports.

`POST /api/plan` accepts:

```json
{
  "message": "Plan a 7 day Japan trip from Shanghai with an 80000 CNY budget.",
  "refinement_context": {
    "mode": "refine",
    "previous_request": "Plan a 7 day Japan trip from Shanghai with an 80000 CNY budget.",
    "followup_message": "Upgrade food within the current budget.",
    "target_categories": ["food"],
    "allow_total_budget_increase": false,
    "max_activity_radius_km": 10
  }
}
```

`refinement_context` is optional. Normal first-pass planning remains compatible with a plain `message`.

## Refinement Behavior

The frontend detects follow-up upgrade intent after a successful report has been generated. Supported upgrade categories are:

- `food`
- `attractions`
- `flight`
- `hotel`

Refinement is based on the latest successful plan, not the first version of the conversation. The system stores the latest planning baseline, including total budget and selected total spend. If the user previously increased the budget, that new total becomes the active planning budget for subsequent refinements.

If a category is already at the current budget ceiling, the frontend asks whether the user wants to increase the total budget. If the user declines, the existing plan is preserved. If a generated refinement would reduce total selected spend compared with the previous successful plan, the backend rejects that result and keeps the previous plan.

For food and attractions, the frontend asks whether the user accepts farther high-spec options before expanding the activity radius beyond the default local range.

## Output Files

- `travel_reports/`: generated HTML itinerary reports.
- `debug_reports/`: parser/debug output.
- `client_itineraries_*`: batch-generated itinerary artifacts.

These files are useful during local development and validation. Treat generated reports and debug artifacts as runtime outputs unless they are intentionally being used as fixtures.

## Development Notes

- The Python backend is designed for local development and demos, not as a hardened production API server.
- The planning pipeline performs live searches when real API keys are configured.
- Budget feasibility checks may request a higher budget or recommend relaxing trip constraints when fixed costs exceed the user's stated budget.
- Monetary values in reports are formatted as whole-number amounts.
- The itinerary engine includes safeguards for transfer placement, hotel transitions, meal windows, duplicate POIs, and overlapping event timelines.
