# Beta-Binomial 批次更新与预测后端

本项目从空目录实现一个本地 FastAPI 后端。观测值由成功数和失败数组成，先验参数 `alpha`、`beta` 必须为严格正的有限数值；所有计数必须是非负整数。数据默认保存在本地 SQLite 文件 `data/beta_binomial.sqlite3`，测试使用内存 SQLite 和固定合成夹具。

## 算法假设

1. 先验为 `p ~ Beta(alpha, beta)`。
2. 每个批次的观测在给定 `p` 时服从 Binomial，成功数和失败数是 Beta 模型的充分统计量。
3. 观测到 `s` 个成功、`f` 个失败后，后验为：

   `p | data ~ Beta(alpha + s, beta + f)`
4. 批次修订不删除历史。旧批次版本标记为 `superseded`，新版本标记为 `active`；当前后验只汇总 active 批次，因此修订等于撤回旧的 `(s,f)` 后加入新的 `(s',f')`。
5. 未来 `n` 次试验的预测分布为 Beta-Binomial：

   `P(K=k) = C(n,k) B(alpha_post+k, beta_post+n-k) / B(alpha_post, beta_post)`

普通参数范围使用 SciPy 的 `betaln` 计算 beta 归一化常数差值，避免直接乘组合数和 beta 密度。超大参数导致 `gammaln` 溢出时，统计内核切换到上升阶乘的 log-space 求和；预测概率统一经过 log-sum-exp 和显式归一化。

`credible_interval` 是 Beta 后验等尾可信区间；`confidence_interval` 是观测比例的 Wilson 频率置信区间。二者名称和解释在响应契约中分开。无观测时没有 Wilson 区间，响应字段为 `null`。

## 模块关系

- `app/domain.py`：正先验、非负整数、置信水平、SQLite 计数上限和预测数组上限。
- `app/errors.py`：输入、状态、资源、计算四类领域错误及 HTTP 状态码。
- `app/stats.py`：Beta 后验、后验矩、可信区间、Wilson 区间、稳定 Beta-Binomial PMF。
- `app/storage.py`：SQLite 建表、运行创建、不可变批次版本、运行版本和事务。
- `app/schemas.py`：FastAPI/Pydantic 请求和响应契约。
- `app/service.py`：统计内核与存储之间的用例编排。
- `app/api.py`：HTTP 路由、依赖注入和统一错误响应。
- `experiments/`：固定合成夹具和确定性复现实验，不依赖被测核心生成参考答案。
- `tests/`：统计、存储修订、API、错误类别和实验报告测试；`tests/replay_logger.py` 输出 JSONL 证据。

## 错误契约

- `invalid_input`，HTTP 422：先验非正/非有限、计数不是非负整数、区间水平不在 `(0,1)`。
- `state_conflict`，HTTP 409：运行 ID 已存在、重复提交批次；HTTP 404：运行或待修订批次不存在。
- `resource_exhausted`，HTTP 413：预测点数超过 10000；HTTP 503：SQLite 计数超过 64 位有符号整数或本地数据库资源不可用。
- `computation_failure`，HTTP 500：数值内核出现非有限结果或无法归一化。

## HTTP 快速验证

启动服务：

```bash
BETA_BINOMIAL_DB=data/local-dev.sqlite3 python3 -m uvicorn app.api:app --reload
```

示例请求：

```bash
curl -s -X POST http://127.0.0.1:8000/runs \
  -H 'content-type: application/json' \
  -d '{"run_id":"demo","alpha":1,"beta":1}'

curl -s -X POST http://127.0.0.1:8000/runs/demo/batches/A \
  -H 'content-type: application/json' \
  -d '{"successes":3,"failures":1,"reason":"initial"}'

curl -s -X PUT http://127.0.0.1:8000/runs/demo/batches/A \
  -H 'content-type: application/json' \
  -d '{"successes":1,"failures":1,"reason":"correction"}'

curl -s http://127.0.0.1:8000/runs/demo/posterior

curl -s -X POST http://127.0.0.1:8000/runs/demo/predictions \
  -H 'content-type: application/json' \
  -d '{"future_n":3,"confidence_level":0.95}'

curl -s http://127.0.0.1:8000/runs/demo/versions
```

## 本地验证命令

运行全部测试：

```bash
python3 -m pytest
```

预期：所有测试通过。测试末尾输出 `Replay log: .../logs/pytest-<运行编号>.jsonl`。该日志包含运行编号、测试节点、关键中间状态、参考值、判断理由和失败堆栈。

运行确定性复现实验：

```bash
python3 -m experiments.run_experiment
```

预期：命令输出 `"all_passed": true`，报告写入 `artifacts/experiments/experiment_report.json`。报告检查：

- 分批后验与一次合并后验完全相同。
- 预测概率和为 1，并与独立 `scipy.stats.betabinom` 参考一致。
- 批次修订后的总成功/失败数等于撤回旧批次后加入新批次。
- 版本序列包含 `created`、`batch_submitted`、`batch_revised`。
- 100 万成功、200 万失败等大计数下结果保持有限。

## 依赖版本

本项目按本机约定直接使用全局 Python 3，不创建虚拟环境。当前验证环境：

- Python 3.12
- fastapi 0.141.1
- numpy 2.5.3
- scipy 1.18.1
- pydantic 2.13.5
- pytest 9.1.1
- uvicorn 0.54.0
- httpx 0.28.1
- SQLite：Python 标准库 `sqlite3`
