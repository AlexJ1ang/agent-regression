# 固定多轮任务测试

这是第一版可运行能力：固定消息序列、真实会话传递、工具参数检查、结构化结果检查、独立状态观察、逐轮保存和离线重新检查。模拟用户、自动任务生成以及直接读取 Dify 节点记录尚未实现。

## 定义任务

参见 `examples/tasks/suite.yaml`。任务由 `id`、`goal`、可发送的 `context`、`turns` 和检查条件组成。
`goal` 与预期值仅供测试侧使用，不发送给被测 Agent。每轮只发送当前 `message`、`context` 以及此前响应中的 `conversation_id`。
任务 ID 不可重复；未知字段会报错。没有检查的任务进入 `review`。

检查包括：

| kind | 含义 |
| --- | --- |
| `answer_contains` | 当轮回答必须包含非空字符串 |
| `answer_json` | 将回答作为 JSON，核对 `path` 对应的 `expected` |
| `tool_call` | 至少观察到一次指定工具调用，`arguments` 为必需参数的递归子集 |
| `tool_not_called` | 指定工具在检查范围内不能出现 |
| `state` | 检查独立状态接口返回的对象，使用 `path` 与 `expected` |

`path` 使用点分隔，列表使用索引，例如 `work_orders.0.priority`。数值不支持模糊容差；布尔值和 0/1 不混用。
每轮检查仅观察当轮。任务级检查观察最终回答/状态与整组累计工具调用。
已观察到的调用可作为确定证据；判断整组没有某调用时，需要每轮都有完整工具记录。
检查工具是否出现不等于验证调用顺序或证明整个任务完成。按业务要求添加参数、结果和状态检查。

## 接入已有应用

`--target` 读取独立 YAML 配置。最小配置是 `url`。请求默认如下：

```json
{"query":"当前问题","inputs":{"role":"reader"},"conversation_id":"上一轮服务返回的ID"}
```

首轮不发送 `conversation_id`。可用 `query_field`、`context_field`、`conversation_id_field` 修改字段名，`extra_body` 增加固定字段（例如响应模式）。
响应默认如下：

```json
{"answer":"回答文本","conversation_id":"服务创建的ID","tool_calls":[{"name":"query_energy","arguments":{"device_id":"B","month":"2026-09"}}]}
```

`answer_path`、`conversation_id_path`、`tools_path` 支持点分隔路径。`answer` 必须为字符串，结构化结果应作为 JSON 文本返回。工具记录必须来自实际执行或可信的运行记录；仅观察最终文本的接口无法验证工具调用。
省略 `tool_calls` 表示看不到工具；空数组表示观察到本轮没有工具。无效工具结构会报错。
多轮缺少会话 ID 或中途更换会话 ID 时停止该任务，保存异常并继续其他任务。

认证用 `headers_env: {Authorization: AGENT_AUTHORIZATION}` 引用环境变量。配置填写变量名，值由运行环境注入。不要将令牌放进 URL 或提交到仓库。
`revision` 用于记录目标应用/提示词/模型版本；框架不能替你识别服务背后的部署变化。

可选 `state_url` 指向可信、只读的测试环境观察接口，例如 `http://localhost:8765/state/{conversation_id}`。
状态由独立 GET 请求读取，不采用回答正文或响应中的自述 `state`。真实状态访问及工具暴露需要目标应用提供相应接入，配置不能凭空获得内部信息。

## 结果和继续执行

- `passed`：所有配置检查均通过。
- `failed`：有明确违反检查条件的证据；后续异常不会抹掉已观察到的失败。
- `review`：检查或证据不足。
- `error`：调用或会话执行异常。
- `invalid`：为后续模拟交互的无效用例保留；当前静态配置错误在执行前报错。

只有所有任务均为 `passed`，命令才返回 0。每轮完整消息、回答、工具参数、状态、检查和端到端耗时均保存到 `run.json`，文件以原子替换写入。

```bash
# 跳过完成的任务，未完成任务从新会话重跑
agent-regression task -s examples/tasks/suite.yaml -t examples/tasks/target.yaml \
  -o harness_output/task-demo --resume

# 不联网，用当前检查重新评估保存的响应
agent-regression task -s examples/tasks/suite.yaml -t examples/tasks/target.yaml \
  -o harness_output/task-demo --rejudge
```

目标配置和执行输入需要与记录一致；修改检查允许离线重新判定，不能混入一次继续执行中。
新会话不等于重置真实业务数据。涉及工单、订单等动作时，接入方应提供隔离测试数据或幂等处理；本地演示按会话隔离其虚构状态。
这版不承诺恢复任意 Agent 的中间状态。所有记录可能含敏感业务信息，应按业务数据管理。

## 旧版单轮评分的变化

`eval` 命令保留。跳过维度不再抬高总分，断流/请求异常进入 `error`，没有可执行检查进入 `review`。
参考答案的数字集合覆盖已移除；仅有参考答案且没有可用 Judge 或独立业务检查时不能认证通过。
业务通过率为 `passed / (passed + failed)`，待核验和执行异常单独报告；发布检查会拒绝证据不完整的运行。
明确的通过→失败会阻止回归放行，不再因单样本缺乏统计显著性而忽略。
要求 `--gate-max-regressions` 时必须有有效历史结果；首次运行应先不启用该项，检查结果后保存初始基线。
本版改变了分数归一化和通过判定口径。新评分不会直接与旧评分结果比较，应重新执行、复核并建立新基线。Judge 缓存也改为绑定完整请求，旧缓存不会用于认证新评分。
