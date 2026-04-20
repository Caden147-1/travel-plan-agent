import json
import traceback
from datetime import datetime
from pathlib import Path

from router import run_budget_pipeline


TEST_CASES = [
    # --------------------------------------------------
    # A. Basic single-city
    # --------------------------------------------------
    {
        "id": "A01",
        "category": "single_city_basic",
        "description": "单城市，预算充足，要求直飞",
        "query": "I want to travel from Singapore to Tokyo from 2026-04-19 to 2026-04-23 with a budget of 30000 CNY. I prefer direct flights."
    },
    {
        "id": "A02",
        "category": "single_city_basic",
        "description": "单城市，预算适中，无特殊偏好",
        "query": "I want to travel from Beijing to Seoul from 2026-05-01 to 2026-05-05 with a budget of 8000 CNY."
    },
    {
        "id": "A03",
        "category": "single_city_basic",
        "description": "单城市，只给国家不指定城市",
        "query": "I want to travel from Shanghai to Japan from 2026-06-10 to 2026-06-15 with a budget of 15000 CNY."
    },

    # --------------------------------------------------
    # B. Flight preference
    # --------------------------------------------------
    {
        "id": "B01",
        "category": "flight_direct",
        "description": "明确要求直飞",
        "query": "I want to travel from Singapore to Tokyo from 2026-04-19 to 2026-04-23 with a budget of 12000 CNY. I want direct flights."
    },
    {
        "id": "B02",
        "category": "flight_comfort",
        "description": "要求更舒适航班",
        "query": "My friend and I want to travel from Hong Kong to Seoul from 2026-07-01 to 2026-07-06 with a budget of 18000 CNY. We prefer comfort flights."
    },
    {
        "id": "B03",
        "category": "flight_cheap",
        "description": "要求便宜航班",
        "query": "I want to travel from Guangzhou to Bangkok from 2026-08-10 to 2026-08-15 with a budget of 5000 CNY. I want cheap flights."
    },

    # --------------------------------------------------
    # C. Hotel preference
    # --------------------------------------------------
    {
        "id": "C01",
        "category": "hotel_budget",
        "description": "budget hotel",
        "query": "I want to travel from Shenzhen to Kuala Lumpur from 2026-06-01 to 2026-06-06 with a budget of 6000 CNY. I want budget hotels."
    },
    {
        "id": "C02",
        "category": "hotel_luxury",
        "description": "luxury hotel with enough budget",
        "query": "I want to travel from Singapore to Tokyo from 2026-04-19 to 2026-04-23 with a budget of 30000 CNY. I want five star hotels."
    },
    {
        "id": "C03",
        "category": "hotel_luxury_low_budget",
        "description": "luxury hotel but obviously under budget",
        "query": "I want to travel from Shanghai to Tokyo from 2026-04-19 to 2026-04-23 with a budget of 3000 CNY. I want five star hotels and direct flights."
    },
    {
        "id": "C04",
        "category": "hotel_central",
        "description": "central hotel preference",
        "query": "My teacher and I want to travel from Hong Kong to United States from 2026-10-12 to 2026-10-22 with a budget of 5000 CNY. We want a centrally located premium hotel."
    },
    {
        "id": "C05",
        "category": "hotel_clean",
        "description": "clean hotel preference",
        "query": "I want to travel from Beijing to Osaka from 2026-05-10 to 2026-05-15 with a budget of 12000 CNY. I want clean hotels."
    },

    # --------------------------------------------------
    # D. Food preference
    # --------------------------------------------------
    {
        "id": "D01",
        "category": "food_local",
        "description": "local food only",
        "query": "I want to travel from Guangzhou to Bangkok from 2026-07-10 to 2026-07-15 with a budget of 10000 CNY. I want local food."
    },
    {
        "id": "D02",
        "category": "food_fine_dining",
        "description": "fine dining preference",
        "query": "I want to travel from Singapore to Paris from 2026-09-01 to 2026-09-07 with a budget of 35000 CNY. I want fine dining."
    },
    {
        "id": "D03",
        "category": "food_street_food",
        "description": "street food preference",
        "query": "I want to travel from Shenzhen to Bangkok from 2026-08-10 to 2026-08-14 with a budget of 7000 CNY. I love street food."
    },
    {
        "id": "D04",
        "category": "food_conflict",
        "description": "local food + low budget / check fallback behavior",
        "query": "I want to travel from Singapore to Tokyo from 2026-04-19 to 2026-04-23 with a budget of 5000 CNY. I want local food and direct flights."
    },

    # --------------------------------------------------
    # E. Attraction preference
    # --------------------------------------------------
    {
        "id": "E01",
        "category": "attraction_historical",
        "description": "historical attractions",
        "query": "I want to travel from Hong Kong to Rome from 2026-09-10 to 2026-09-17 with a budget of 28000 CNY. I enjoy historical places and museums."
    },
    {
        "id": "E02",
        "category": "attraction_nature",
        "description": "nature attractions",
        "query": "My friend and I want to travel from Seoul to Switzerland from 2026-04-19 to 2026-04-24 with a budget of 25000 CNY. I love mountains and natural scenery."
    },
    {
        "id": "E03",
        "category": "attraction_shopping",
        "description": "shopping preference",
        "query": "I want to travel from Shanghai to Singapore from 2026-05-01 to 2026-05-05 with a budget of 15000 CNY. I want shopping."
    },
    {
        "id": "E04",
        "category": "attraction_art",
        "description": "art preference",
        "query": "I want to travel from Beijing to Paris from 2026-07-01 to 2026-07-08 with a budget of 30000 CNY. I enjoy art museums and galleries."
    },
    {
        "id": "E05",
        "category": "attraction_heavy",
        "description": "many attractions / packed itinerary",
        "query": "I want to travel from Singapore to Tokyo from 2026-04-19 to 2026-04-23 with a budget of 18000 CNY. I want a packed itinerary with many attractions."
    },

    # --------------------------------------------------
    # F. Multi-city
    # --------------------------------------------------
    {
        "id": "F01",
        "category": "multi_city_basic",
        "description": "multi-city in one country",
        "query": "I want to travel from Tokyo to Osaka for 3 days and Kyoto for 2 days from 2026-06-01 to 2026-06-06 with a budget of 12000 CNY."
    },
    {
        "id": "F02",
        "category": "multi_city_us",
        "description": "multi-city US",
        "query": "My friend and I want to travel from Hong Kong to New York for 4 days, Los Angeles for 4 days, and San Francisco for 3 days from 2026-10-12 to 2026-10-22 with a budget of 5000 CNY. We enjoy historical places and art galleries. We want a centrally located premium hotel."
    },
    {
        "id": "F03",
        "category": "multi_city_fallback_airport",
        "description": "multi-city with non-airport city",
        "query": "My friend and I want to travel from Seoul to Switzerland from 2026-04-19 to 2026-04-24 with a budget of 25000 CNY. I love mountains and natural scenery."
    },

    # --------------------------------------------------
    # G. Budget stress tests
    # --------------------------------------------------
    {
        "id": "G01",
        "category": "budget_extremely_low",
        "description": "极低预算",
        "query": "I want to travel from Hong Kong to New York from 2026-10-12 to 2026-10-22 with a budget of 3000 CNY."
    },
    {
        "id": "G02",
        "category": "budget_low_conflict",
        "description": "低预算 + 高要求",
        "query": "I want to travel from Shanghai to Tokyo from 2026-04-19 to 2026-04-23 with a budget of 3000 CNY. I want five star hotels and direct flights."
    },
    {
        "id": "G03",
        "category": "budget_high",
        "description": "高预算",
        "query": "I want to travel from Singapore to Tokyo from 2026-04-19 to 2026-04-23 with a budget of 50000 CNY. I want direct flights and luxury hotels."
    },

    # --------------------------------------------------
    # H. Missing / incomplete info
    # --------------------------------------------------
    {
        "id": "H01",
        "category": "missing_dates",
        "description": "缺日期",
        "query": "I want to travel from Beijing to Tokyo with a budget of 12000 CNY."
    },
    {
        "id": "H02",
        "category": "missing_budget",
        "description": "缺预算",
        "query": "I want to travel from Beijing to Tokyo from 2026-05-01 to 2026-05-05."
    },
    {
        "id": "H03",
        "category": "missing_origin",
        "description": "缺出发地",
        "query": "I want to travel to Tokyo from 2026-05-01 to 2026-05-05 with a budget of 10000 CNY."
    },

    # --------------------------------------------------
    # I. Parsing edge cases
    # --------------------------------------------------
    {
        "id": "I01",
        "category": "relative_date_text",
        "description": "相对文本日期 / 模糊日期格式",
        "query": "I want to travel from Singapore to Tokyo on 4.19-4.23 with a budget of 30000 CNY and direct flights."
    },
    {
        "id": "I02",
        "category": "two_people_parse",
        "description": "两人出行",
        "query": "My friend and I want to travel from Singapore to Tokyo from 2026-04-19 to 2026-04-23 with a budget of 30000 CNY. We want direct flights."
    },
    {
        "id": "I03",
        "category": "teacher_and_i",
        "description": "teacher and I 人数测试",
        "query": "My teacher and I want to travel from Hong Kong to Tokyo from 2026-05-01 to 2026-05-06 with a budget of 18000 CNY."
    },
]


def safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return cur if cur is not None else default


def summarize_result(case, result):
    summary = {
        "id": case["id"],
        "category": case["category"],
        "description": case["description"],
        "success": result.get("success"),
        "stage": result.get("stage"),
        "origin": safe_get(result, "trip_summary", "origin"),
        "destination": safe_get(result, "trip_summary", "destination", default=[]),
        "num_people": safe_get(result, "trip_summary", "num_people"),
        "duration_days": safe_get(result, "trip_summary", "duration_days"),
        "budget_total": safe_get(result, "trip_summary", "budget", "total"),
        "budget_currency": safe_get(result, "trip_summary", "budget", "currency"),
        "warnings": result.get("warnings", []),
        "flight_total_status": safe_get(result, "budget_fit_status", "flight_total"),
        "hotel_statuses": [
            x.get("hotel_selection_status")
            for x in result.get("city_selection_summary", [])
            if isinstance(x, dict)
        ],
        "restaurant_statuses": [
            x.get("restaurant_selection_status")
            for x in result.get("city_selection_summary", [])
            if isinstance(x, dict)
        ],
        "poi_statuses": [
            x.get("poi_selection_status")
            for x in result.get("city_selection_summary", [])
            if isinstance(x, dict)
        ],
        "selected_outbound_count": len(result.get("selected_outbound_flights", [])),
        "selected_intercity_count": len(result.get("selected_intercity_flights", [])),
        "selected_inbound_count": len(result.get("selected_inbound_flights", [])),
        "selected_hotels_count": len(result.get("selected_hotels", [])),
        "selected_restaurants_count": len(result.get("selected_restaurants", [])),
        "selected_pois_count": len(result.get("selected_pois", [])),
        "budget_check": result.get("budget_check", {}),
        "agent2_budget_status": safe_get(result, "agent2_summary", "budget_status"),
    }
    return summary


def print_pretty_summary(summary):
    print("=" * 90)
    print(f"[{summary['id']}] {summary['category']} | {summary['description']}")
    print(f"success={summary['success']} | stage={summary['stage']}")
    print(f"origin={summary['origin']} -> destination={summary['destination']}")
    print(
        f"num_people={summary['num_people']} | duration_days={summary['duration_days']} | "
        f"budget={summary['budget_total']} {summary['budget_currency']}"
    )
    print(
        f"flights(outbound/intercity/inbound)="
        f"{summary['selected_outbound_count']}/"
        f"{summary['selected_intercity_count']}/"
        f"{summary['selected_inbound_count']}"
    )
    print(
        f"hotels={summary['selected_hotels_count']} | "
        f"restaurants={summary['selected_restaurants_count']} | "
        f"pois={summary['selected_pois_count']}"
    )
    print(f"flight_total_status={summary['flight_total_status']}")
    print(f"hotel_statuses={summary['hotel_statuses']}")
    print(f"restaurant_statuses={summary['restaurant_statuses']}")
    print(f"poi_statuses={summary['poi_statuses']}")
    print(f"agent2_budget_status={summary['agent2_budget_status']}")
    if summary["warnings"]:
        print("warnings:")
        for w in summary["warnings"]:
            print(f"  - {w}")
    else:
        print("warnings: []")


def _build_client_itinerary_payload(case, result):
    trip_summary = result.get("trip_summary", {}) if isinstance(result, dict) else {}
    itinerary = result.get("itinerary", {}) if isinstance(result, dict) else {}

    return {
        "case_id": case.get("id"),
        "case_description": case.get("description"),
        "query": case.get("query"),
        "trip_summary": trip_summary,
        "itinerary": {
            "success": itinerary.get("success"),
            "stage": itinerary.get("stage"),
            "itinerary_summary": itinerary.get("itinerary_summary", {}),
            "daily_plan": itinerary.get("daily_plan", []),
            "cost_rollup": itinerary.get("cost_rollup", {}),
            "warnings": itinerary.get("warnings", []),
        },
    }


def _save_client_itinerary_files(results, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    index = []
    for record in results:
        case = record.get("case", {})
        result = record.get("result", {})
        case_id = case.get("id", "unknown")

        payload = _build_client_itinerary_payload(case, result)
        file_path = output_dir / f"{case_id}_itinerary.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        index.append({
            "case_id": case_id,
            "file": file_path.name,
            "itinerary_success": payload.get("itinerary", {}).get("success"),
        })

    index_path = output_dir / "index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    return output_dir


def run_all_tests(save_json=True):
    results = []
    failures = []

    for case in TEST_CASES:
        print(f"\nRunning {case['id']} ...")
        try:
            result = run_budget_pipeline(case["query"])
            summary = summarize_result(case, result)

            record = {
                "case": case,
                "result": result,
                "summary": summary,
            }
            results.append(record)

            print_pretty_summary(summary)

            if not result.get("success", False):
                failures.append({
                    "id": case["id"],
                    "reason": result.get("error", "unknown error"),
                })

        except Exception as e:
            failures.append({
                "id": case["id"],
                "reason": str(e),
                "traceback": traceback.format_exc(),
            })
            print("=" * 90)
            print(f"[{case['id']}] EXCEPTION")
            print(str(e))
            print(traceback.format_exc())

    if save_json:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"batch_test_results_{timestamp}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nSaved full results to: {output_path}")

        itinerary_dir = Path(f"client_itineraries_{timestamp}")
        saved_dir = _save_client_itinerary_files(results, itinerary_dir)
        print(f"Saved client itinerary files to: {saved_dir}")

    print("\n" + "#" * 90)
    print("FINAL SUMMARY")
    print(f"Total test cases: {len(TEST_CASES)}")
    print(f"Failed / exception cases: {len(failures)}")

    if failures:
        print("Failure list:")
        for item in failures:
            print(f"  - {item['id']}: {item['reason']}")

    # quick issue scan
    print("\nPotential issue scan:")
    for record in results:
        case = record["case"]
        summary = record["summary"]

        issue_flags = []

        if summary["success"] is not True:
            issue_flags.append("pipeline_failed")

        if summary["selected_outbound_count"] == 0 and "missing" not in case["category"]:
            issue_flags.append("no_outbound")

        if summary["selected_hotels_count"] == 0 and summary["success"]:
            issue_flags.append("no_hotels")

        if summary["selected_restaurants_count"] == 0 and summary["success"]:
            issue_flags.append("no_restaurants")

        if summary["selected_pois_count"] == 0 and summary["success"]:
            issue_flags.append("no_pois")

        if summary["flight_total_status"] == "over_budget" and not summary["warnings"]:
            issue_flags.append("flight_over_budget_but_no_warning")

        if case["category"] == "two_people_parse" and summary["num_people"] != 2:
            issue_flags.append("num_people_parse_issue")

        if issue_flags:
            print(f"  - {case['id']}: {issue_flags}")


if __name__ == "__main__":
    run_all_tests(save_json=True)
