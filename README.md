# AIGC2 Travel Planning Project

## Run the web app

```bash
python main.py
```

This starts:

- Python API: `http://127.0.0.1:8000`
- Next frontend: `http://127.0.0.1:3000`

If `front-end/node_modules` is missing, `main.py` runs `pnpm install` when pnpm is available, otherwise `npm install`. The browser opens automatically after the frontend is ready.

Frontend flow:

1. Enter a travel request in the chat box.
2. The frontend sends `POST /api/plan` to the Python API.
3. The API calls `router.run_budget_pipeline(...)`.
4. The generated HTML report is served under `/reports/<filename>.html`.

## 项目概览
本项目当前是一个 4 阶段旅行规划流水线：
1. Agent1：需求解析（自然语言 -> 结构化）
2. Agent2：预算规划（分类预算 + 城市级预算）
3. Agent3：资源筛选（航班/酒店/餐厅/景点）
4. Agent4：行程编排（按天输出 daily plan）

主流程入口是 `router.py` 的 `run_budget_pipeline()`。

## 运行环境
- Python 3.10+
- 依赖包：
  - `requests`
  - `python-dotenv`

安装：
```bash
pip install requests python-dotenv
```

## 环境变量
请在项目根目录 `.env` 配置：

```env
DASHSCOPE_API_KEY=...
TRAVEL_LLM_MODEL=qwen-plus
DASHSCOPE_CHAT_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
GOOGLE_MAPS_API_KEY=...
SEARCHAPI_API_KEY=...
ENABLE_HOTEL_LLM_ADJUSTMENT=true

HEIGIT_API_KEY=...
HEIGIT_BASE_URL=https://api.openrouteservice.org
```

说明：
- Agent1/Agent2 依赖 DashScope 兼容接口。
- Agent3 依赖 Google Places + SearchAPI。
- Agent4 跨城 `drive` 时长会优先走 HEIGIT(openrouteservice) 路线 API，失败时回退默认时长。

## 如何运行

### 1) 单条请求完整流程（推荐）
```bash
python router.py
```
- 读取 `router.py` 里的 `test_input`
- 执行 Agent1 -> Agent2 -> Agent3 -> Agent4
- 控制台输出最终 JSON（包含 `itinerary` 字段）

### 2) 批量回归测试
```bash
python batch_test_router.py
```
会生成两类文件：
- `batch_test_results_YYYYMMDD_HHMMSS.json`：所有测试 case 的完整汇总结果
- `client_itineraries_YYYYMMDD_HHMMSS/`：客户展示文件夹
  - 每个 case 一个文件：`<CASE_ID>_itinerary.json`
  - `index.json`：文件索引

### 3) 代码内调用
```python
from router import run_budget_pipeline

query = "My wife and I want to travel from Shanghai to Tokyo and Osaka from 2026-05-01 to 2026-05-07 with a budget of 26000 CNY."
result = run_budget_pipeline(query)
print(result["itinerary"])  # Agent4 行程结果
```

## 当前数据/构建脚本

### 城市默认计划数据
- 输入：`cities15000.txt`、`countryInfo.txt`
- 构建脚本：`build_country_default_city_plan.py`
- 输出：
  - `country_default_city_plan.json`
  - `country_default_city_plan.py`

重建命令：
```bash
python build_country_default_city_plan.py
```

### 机场索引数据
- 输入：`airports.csv`
- 构建脚本：`airport.py`
- 输出：`global_airports.json`

重建命令：
```bash
python airport.py
```

## 核心文件说明

### 主流程
- `router.py`：主编排，已接入 Agent4
- `travel_requirement_parser_llm_agent.py`：Agent1
- `destination.py`：城市计划增强
- `travel_budget_agent.py`：Agent2
- `agent3_real.py`：Agent3

### 行程规划
- `itinerary_agent.py`：Agent4 主实现
- `planner_schema.py`：Agent4 基础 schema/常量
- `itinerary_rules.py`：规则层
- `itinerary_cluster.py`：点位聚合
- `itinerary_builder.py`：行程构建
- `itinerary_validator.py`：约束与汇总校验

### 测试
- `batch_test_router.py`：批量测试入口 + 客户行程文件导出

### 规划说明
- `行程规划_v1.md`

## 现阶段实现特征
- Agent4 已在 `router.py` 自动执行。
- 若 Agent4 异常，主流程仍返回 Agent1~3 结果，并附带 `itinerary` 错误信息。
- 当前城市内短距通勤费用仍按规则常量（默认 200/天）计入。
- 当前跨城驾驶时长优先使用 HEIGIT 路线 API。
