# Agent Regression

在修改 Agent 的提示词或流程后，重新运行一组固定问题，检查原本正常的回答和工具调用是否出现退步。

当前版本通过 HTTP/SSE 接口接入 Agent。用 YAML 描述请求、返回事件和每条问题的预期，再用命令运行测试、保存结果并与上次运行比较。

> 仓库中的服务地址、城市、工具名和用例均为虚构示例，不包含真实客户或设备数据。

## 能做什么

- 重新运行已有问题，检查回答是否满足预期、是否调用了需要的工具。
- 保存每次回答、工具调用和响应时间，比较修改前后的测试结果。
- 单独重跑失败问题，查看新旧回答、工具调用和耗时的差异。
- 生成 HTML 和 JSON 报告，保留每次运行的原始记录。
- 按项目要求设置通过条件，供发布前检查使用。

重复运行、统计分析、模型辅助评分和并发压测可按需启用。

## 固定多轮业务任务（试用版）

`task` 命令按顺序运行一项任务的多轮消息，使用服务返回的会话 ID 继续交互。
每轮可检查工具参数、结构化回答和独立读取的测试环境状态。缺少证据会标为
`review`，调用异常标为 `error`；两者都不会被当作通过。

先启动本地演示服务（无需模型或 API Key）：

```bash
python examples/task_demo.py
```

在另一个终端运行：

```bash
agent-regression task --suite examples/tasks/suite.yaml \
  --target examples/tasks/target.yaml --output harness_output/task-demo
```

打开输出目录的 `report.html` 查看每轮对话、工具参数、检查依据和耗时；
`run.json` 保存完整记录。示例涵盖设备/月度查询和确认后创建工单，两组各三轮。
它们是用于验证框架的确定性测试替身，不代表真实模型效果。

将服务改为 `python examples/task_demo.py --fault wrong-device`（先停止旧服务），
再选择新的输出目录执行，可以验证错误设备查询能否被识别。另有 `wrong-month`
和 `no-tools` 两种注入模式，后者应进入待核验。

`--resume` 跳过已完成任务，未完成任务从新会话开始；`--rejudge` 使用已保存的
响应重新检查，不调用 Agent。重新检查允许修改检查条件，但不允许改变原执行输入或目标。
详细接入、状态含义和限制见 [任务测试说明](docs/task-testing.md)。

## 安装

需要 Python 3.11、3.12 或 3.13。

```bash
git clone https://github.com/AlexJ1ang/agent-regression.git
cd agent-regression
python -m venv .venv
```

激活虚拟环境：

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

安装：

```bash
python -m pip install -e .

# 参与开发时安装测试、lint 和类型检查工具
python -m pip install -e ".[dev]"
```

## 五分钟接入自己的 Agent

### 1. 复制示例配置

```bash
cp examples/config.yaml config.local.yaml
```

Windows PowerShell：

```powershell
Copy-Item examples/config.yaml config.local.yaml
```

`config.local.yaml` 已被 Git 忽略，可以安全地放置本地地址或认证请求头。仍建议优先使用短期令牌，并避免把任何密钥写入可提交文件。

### 2. 映射 HTTP 请求

假设你的接口接收：

```json
{
  "message": "用户问题",
  "conversation_id": "动态会话 ID",
  "metadata": {"locale": "zh-CN"}
}
```

把 `target.request_adapter.body_template` 改成：

```yaml
target:
  base_url: "http://localhost:8000"
  chat_path: "/api/agent/stream"
  method: POST
  request_adapter:
    body_template:
      message: "{question}"
      conversation_id: "{session_id}"
      metadata: "{context}"
```

可用占位符：

| 占位符 | 来源 |
|---|---|
| `{question}` | 当前测试问题 |
| `{case_id}` | 用例 ID |
| `{context}` | 当前用例的完整 context 对象 |
| `{session_id}` | 每次执行自动生成 |
| `{user_id}` | 根据用例 ID 自动生成 |
| `{任意字段}` | 从当前用例的 `context` 中读取同名字段 |

模板可以包含嵌套对象和数组。若接口需要固定请求头，可写入 `extra_headers`：

```yaml
request_adapter:
  extra_headers:
    X-Client: "agent-regression"
```

认证头请只放在被忽略的 `config.local.yaml` 中，不要提交真实令牌。

也可以用环境变量覆盖常用连接项：

```bash
HARNESS_BASE_URL=http://localhost:9000
HARNESS_CHAT_PATH=/v2/chat/stream
HARNESS_JUDGE_API_KEY=your-key
HARNESS_JUDGE_API_BASE=https://api.openai.com/v1
HARNESS_JUDGE_MODEL=gpt-4o-mini
```

### 3. 映射 SSE 协议

框架期望标准 SSE 文本流，但事件名和 JSON 字段可以自由映射。例如服务返回：

```text
event: token
data: {"text":"Hello"}

event: function_call
data: {"function":"get_weather","args":{"city":"Example City"},"id":"call-1"}

event: finished
data: {"trace_id":"trace-1"}
```

对应配置：

```yaml
target:
  sse_mapping:
    answer_event: "token"
    answer_content_field: "text"
    tool_call_event: "function_call"
    tool_name_field: "function"
    tool_args_field: "args"
    tool_call_id_field: "id"
    done_event: "finished"
    error_event: "error"
    error_message_field: "message"
```

没有结束事件的流会被标记为 `stream_interrupted`。未映射的事件仍会进入 trace，但不参与对应指标计算。

### 4. 为自己的项目编写测试集

复制 `examples/suites/smoke.yaml`，然后替换问题、上下文和期望：

```yaml
id: my-agent-regression
name: My Agent Regression Suite

cases:
  - id: weather-001
    question: "What is the weather in Example City today?"
    context:
      locale: "en-US"
    expectations:
      expected_intents:
        main: "weather_query"
      expected_tools:
        - "get_weather"
      expected_keywords:
        - "Example City"
      expected_patterns:
        - "-?\\d+\\s*°C"
      is_open_ended: false
```

期望字段：

| 字段 | 用途 |
|---|---|
| `expected_intents` | 检查状态事件中识别出的意图 |
| `expected_tools` | 检查工具是否被调用，不要求调用顺序 |
| `expected_keywords` | 检查最终回答是否包含关键词 |
| `expected_patterns` | 用正则检查最终回答 |
| `reference_answer` | 供 Judge 或业务检查核对；仅凭数值出现不会判定正确 |
| `is_open_ended` | 为 `true` 时启用 LLM-as-Judge |

测试集支持 YAML、JSON 和 CSV。业务数据应使用虚构值，真实回归用例请放在私有仓库或被 Git 忽略的文件中。

### 5. 运行评测

```bash
agent-regression eval \
  --config config.local.yaml \
  --suite examples/suites/smoke.yaml
```

默认生成 JSON、HTML、trace 和基线：

```text
harness_output/
├── report_<run_id>.json
├── report_<run_id>.html
├── baselines/latest.json
├── traces/<run_id>/<case_id>.json
└── judge_cache/
```

`harness_output` 可能包含完整问题、回答、请求体和工具参数，因此默认被 Git 忽略。

## LLM-as-Judge

确定性的行为优先使用规则评分；只有无法通过规则准确判断的开放式回答才建议启用 Judge。

```yaml
eval:
  scorers_enabled: [intent, tool, fact, llm_judge]
  judge:
    enabled: true
    model: "gpt-4o-mini"
    pass_threshold: 3.0
    cot_enabled: true
    consistency_check: false
```

设置密钥后运行：

```bash
export HARNESS_JUDGE_API_KEY="your-key"
agent-regression eval -c config.local.yaml -s suites/regression.yaml
```

PowerShell 使用 `$env:HARNESS_JUDGE_API_KEY = "your-key"`。Judge API 需兼容 OpenAI Chat Completions 接口。

## 性能压测

```bash
agent-regression load \
  --config config.local.yaml \
  --suite examples/suites/smoke.yaml \
  --concurrency 10 \
  --duration 60
```

请只对你拥有或明确获准测试的服务执行压测，并先在测试环境确认容量限制。

## 回放与报告

```bash
# 查看已有基线和 trace run
agent-regression list

# 回放单个用例并生成回答、工具和耗时 diff
agent-regression replay -c config.local.yaml -s suites/regression.yaml \
  --run-id <run_id> --case-id <case_id>

# 从 JSON 重新生成 HTML
agent-regression report -i harness_output/report_<run_id>.json

# 预览将被清理的旧 trace；加 --yes 才实际删除
agent-regression prune --keep 20
```

## CI 质量门禁

```bash
agent-regression eval \
  -c config.local.yaml \
  -s suites/regression.yaml \
  --gate \
  --gate-min-score 0.70 \
  --gate-min-pass-rate 0.90 \
  --gate-max-regressions 0
```

任一门禁条件失败时命令返回非零退出码。仓库中的 `.github/workflows/ci.yml` 用于验证框架自身；把 Agent 回归评测加入你的项目 CI 时，还需要启动被测服务并安全注入配置和密钥。

## 开发

```bash
pytest
ruff check .
mypy src
```

## 数据安全

- 不要提交 `harness_output`、本地配置、API Key 或访问令牌。
- trace 会保存完整请求、SSE 事件、回答和工具参数；按敏感业务数据管理。
- 对公开示例使用虚构的人名、组织、设备、账户和服务地址。
- 在把真实生产数据用于 Judge 前，确认数据处理与合规要求。

## License

Apache License 2.0，详见 [LICENSE](LICENSE)。
