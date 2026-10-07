# 可配置状态机模型驱动测试后端

这是一个本地、可复现的模型驱动测试（MBT）后端。输入 JSON 状态机模型后，系统生成合法与显式非法操作序列，通过 HTTP 调用被测服务，只有观察结果与模型契约一致时才更新预测模型。证据写入 SQLite，失败时生成缩减后的独立 Python 重放脚本。

## 模块职责

- `mbt_server/core/domain.py`：输入模型、运行请求、计划、观察、运行结果的 Pydantic 契约。
- `mbt_server/core/model_state.py`：预测状态机；调用 SUT 前不推进，观察确认后才推进。
- `mbt_server/core/scheduler.py`：目标、覆盖和随机调度；记录候选、原因、种子 RNG 抽取编号。
- `mbt_server/core/http_adapter.py`：唯一的 SUT HTTP 调用边界，区分状态冲突、输入错误、资源耗尽、计算失败。
- `mbt_server/core/runner.py`：执行与判定内核，负责观察后更新模型、状态保持、失败分类、缩减重验。
- `mbt_server/core/replay.py`：生成仅依赖标准库的最小重放脚本。
- `mbt_server/core/storage.py`：SQLite 证据表 `runs` 和 `steps`。
- `mbt_server/core/engine.py`：编排计划、执行、缩减、脚本和证据存储。
- `mbt_server/api.py`：FastAPI 报告和执行接口。
- `mbt_server/services/reference_app.py`：正确的本地订单服务。
- `mbt_server/services/buggy_app.py`：独立实现的故意有状态错误服务。
- `fixtures/order_model.json`：最小订单生命周期模型和非法操作契约。

## 模型与错误契约

订单状态为 `none -> draft -> submitted -> closed/cancelled`。成功转移要求 2xx 响应并推进模型；非法状态操作要求 HTTP 409 和：

```json
{"category": "state_conflict"}
```

错误类别持久化为稳定字符串：

- `input_error`：模型或请求体不符合契约，对应 400/422。
- `state_conflict`：当前状态不允许该操作，对应 409。
- `resource_exhausted`：本地合成容量耗尽，对应 507。
- `computation_failure`：本地合成计算失败、传输失败或无法解释的 5xx。
- `resource_not_found`：本地资源不存在，对应 404。

## 启动服务

不创建虚拟环境，直接使用题目约定的系统 Python：

```bash
python3 -m uvicorn mbt_server.services.reference_app:app --host 127.0.0.1 --port 8101
python3 -m uvicorn mbt_server.services.buggy_app:app --host 127.0.0.1 --port 8102
MBT_DATABASE_PATH=data/manual_evidence.sqlite3 \
  python3 -m uvicorn mbt_server.api:app --host 127.0.0.1 --port 8000
```

## 执行示例

```bash
python3 scripts/run_example.py reference
python3 scripts/run_example.py buggy
```

故障运行会生成 `data/replay_run_<id>.py`。它独立连接故障服务并以退出码 1 复现：

```bash
python3 data/replay_run_2.py
```

也可直接查询报告：

```bash
curl -s http://127.0.0.1:8000/runs
curl -s http://127.0.0.1:8000/runs/2
curl -s http://127.0.0.1:8000/runs/2/replay
```

## 调度与缩减

`targeted` 为每个显式非法转移生成一个独立资源 episode，并用 BFS 生成最短合法前缀。`coverage` 枚举状态/非法操作对，`random` 使用 `random.Random(seed)`；每次非唯一选择都记录 `rng_draws` 与 `rng_value`。

缩减使用 delta-debug 风格分块删除。每个候选先在模型中重新验证前置条件：必须从 `initial_state` 开始、episode 一致、每步都在当前状态可执行；通过后再创建全新 SUT 资源实际重放，只有错误转移、HTTP 状态和失败类别都一致才接受删除。

## 故意错误

故障服务包含验收要求的三类状态错误：

- 已关闭订单重复关闭，错误返回 200。
- 草稿订单越序关闭，错误返回 200 且底层状态仍是 draft。
- 已关闭订单取消，错误返回 200 且底层状态仍是 closed。

模型还包含已关闭订单越序提交，确保覆盖“越序提交”。

## 测试

```bash
pytest -q
```

测试启动真实 uvicorn HTTP 端口，不断言“接口可调”而已；会检查具体状态码、类别、前后模型状态、SQLite 证据、种子/调度记录、缩减结果、参考不误判和故障必发现。

依赖版本记录在 `requirements.txt`。最近一次验证结果见 `docs/verification.md`。
