# 合成设备寿命：Kaplan-Meier 与 log-rank 后端

这是一个本地 FastAPI 后端，用于对合成设备寿命数据做 Kaplan-Meier 生存曲线和多组 log-rank 比较。它不是医疗设备、临床决策或真实可靠性认证系统；返回结果明确标注为合成/演示用途，不能用于真实医疗结论。

## 模块拆分

- `survival_core/models.py`：寿命观测值，`entry/time/event` 表达左截断、结束时间和事件指示。
- `survival_core/risk.py`：逐时点风险集，固定并列端点规则，展示入风险、事件、删失、风险后人数。
- `survival_core/kaplan_meier.py`：Kaplan-Meier、Greenwood 方差、线性和 log-log 置信区间、中位寿命状态。
- `survival_core/logrank.py`：多组 log-rank，输出观测数、期望数、协方差矩阵、卡方、自由度和 p 值。
- `app/contracts.py`：Pydantic v2 样本契约和错误/响应模型。
- `app/service.py`：版本化分析编排，合并逐组提示、比较失败、请求 trace 和免责声明。
- `app/repository.py`：SQLite 版本化结果与结构化事件日志，schema 版本写在 `PRAGMA user_version`。
- `app/main.py`：FastAPI 路由、请求身份、HTTP 日志、422/404/500 错误外壳。
- `app/synthetic.py`：本地 Weibull 设备寿命 + Uniform 检查删失的合成夹具生成器。
- `scripts/run_reproduction.py`：独立复现实验，输出 JSON 工件。
- `tests/`：手算风险集、并列事件、大量删失、全删失、左截断、边界 CI、API、持久化和 lifelines 参考库对照。

## 统计范围与关键取舍

时间尺度是年龄/设备时间尺度。左截断通过 `entry` 单独声明，风险规则是：

```text
entry < t <= time
```

因此，恰好在 `entry == t` 进入的对象不计入时点 `t` 的风险集；恰好在 `time == t` 结束的对象会留在该时点风险集中。事件与删失发生在同一时刻时，它们共享同一个固定风险集，删失不会在同批事件之前提前离开。

每个组总是返回 `time=0` 基线行；正的纯入风险时点也会列出。点类型包括：

- `baseline`：时间 0，`entry=0` 的对象从这里开始观察。
- `event`：该时点至少有事件；若同时有入风险或删失，事件优先作为点类型。
- `censoring`：只有删失。
- `entry`：只有左截断对象进入观察。

Kaplan-Meier 在事件时点更新：

```text
S(t) = S(t-) * (n - d) / n
Var_G[S(t)] = S(t)^2 * sum(d / (n * (n - d)))
```

删失时点不改变存活率。默认 `greenwood_linear`：用正态近似并裁剪到 `[0,1]`，发生裁剪时 `ci_clipped=true`。也可请求 `log_log`，它使用 `log(-log(S))` 变换，天然保持概率范围。

端点处理：

- 第一个事件前 `S=1`，区间报告为 `[1,1]`。
- 最后一个风险对象也发生事件时，`S=0`，展示退化区间 `[0,0]`，但 Greenwood 方差在该端点不可定义，返回 `greenwood_variance=null`，并给出 `boundary_survival_zero_variance_undefined` 提示。
- 空组没有曲线点、没有中位寿命，`median_status=not_estimable`。
- 无事件或曲线始终高于 0.5 时不伪造中位寿命，`median_survival_time=null`，`median_status=not_reached`。

log-rank 在每个事件时点从同一风险集估计共同事件概率，支持左截断。两组以上使用前 `K-1` 组的协方差矩阵和卡方分布。空组、全组无事件、只有一组、协方差矩阵奇异等情形不返回伪造 p 值；业务性不可比返回 200，并在 `comparison.status/reason` 与 `failures` 或 `notices` 中单列。

## 本地启动

本机已有所需 Python 包，直接使用系统 `python3`，不要创建虚拟环境。

```bash
python3 -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

健康检查：

```bash
curl -s -H 'X-Request-ID: req-health' http://127.0.0.1:8000/health
```

默认数据库路径为 `data/survival.sqlite3`。可用环境变量覆盖：

```bash
SURVIVAL_DB_PATH=data/local-demo.sqlite3 \
SURVIVAL_LOG_LEVEL=DEBUG \
python3 -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

## 示例请求

提交手算夹具：

```bash
curl -s \
  -H 'Content-Type: application/json' \
  -H 'X-Request-ID: req-hand-sample' \
  -d @samples/analysis_request.json \
  http://127.0.0.1:8000/api/v1/analyses
```

响应会包含：

- `analysis_id`、`request_id`、`schema_version`、`kernel_version`、`processing_location`。
- 每组逐时点 `risk_before/entered/events/censored/risk_after/survival`。
- `comparison.status`、卡方和 p 值；不可比时给出原因。
- `trace` 展示风险规则、每组 KM、log-rank、SQLite 持久化等步骤。
- `failures` 和 `notices` 分列失败原因与不确定/边界结论。

回查结果：

```bash
curl -s http://127.0.0.1:8000/api/v1/analyses/<analysis_id>
```

回查可解释事件：

```bash
curl -s http://127.0.0.1:8000/api/v1/analyses/<analysis_id>/events
```

生成确定性合成夹具：

```bash
curl -s \
  -H 'Content-Type: application/json' \
  -d @samples/synthetic_request.json \
  http://127.0.0.1:8000/api/v1/synthetic-dataset
```

## 复现实验

```bash
python3 scripts/run_reproduction.py --output-dir artifacts
```

输出：

- `artifacts/hand_fixture_result.json`：手算输入的逐时点风险、KM 和失败/提示。
- `artifacts/synthetic_dataset.json`：固定种子的合成设备数据。
- `artifacts/synthetic_analysis_result.json`：对合成数据的 log-log CI 和多组 log-rank。
- `artifacts/reproduction_manifest.json`：版本和输出文件清单。

## 测试

```bash
pytest -q
```

交付时也保留了机器可读与文本测试结果：

- `artifacts/pytest-report.xml`
- `artifacts/pytest-output.txt`

测试期望值不是由被测核心自生成。手算测试直接断言每个时点的风险数、事件数、删失数和存活率；参考库测试使用独立安装的 `lifelines==0.30.3` 对照 KM、log-log 区间、两组与多组 log-rank（含 `entry` 左截断参数）。

失败类别覆盖：

- `ENTRY_NOT_BEFORE_TIME`、`NON_POSITIVE_TIME`、`NON_FINITE_VALUE`
- `DUPLICATE_OBSERVATION_ID`、`DUPLICATE_GROUP`、`UNKNOWN_GROUP`
- `empty_group_has_no_median`、`no_events_median_not_reached`
- `boundary_survival_zero_variance_undefined`
- `empty_group_excluded_from_log_rank`、`no_events_in_any_group`

## 不支持项

- 不做 Cox 模型、分层 log-rank、权重 log-rank、竞争风险或复发事件。
- 不做中位寿命置信区间，只报告中位寿命点估计或不可估计/未达到。
- 不提供真实设备或医疗数据，不执行真实业务账号调用。
- 不把无事件、空组或矩阵奇异解释成“组间相同”；这些只返回不可检验/未运行的原因。
