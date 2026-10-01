# 本地共识 ADMM 教学后端

这是一个仅依赖本地合成数据的双进程共识 ADMM 后端，用于可分强凸二次目标

```text
minimize  f_1(z) + f_2(z),
f_i(x_i) = 1/2 x_i' Q_i x_i + q_i' x_i,  Q_i positive definite.
```

项目使用 Python、FastAPI、NumPy、SciPy 和 Pydantic。两个计算进程可以是两个真实的本地子进程：它们通过 HTTP 拉取本轮任务，独立求解本地二次子问题并提交；服务端协调器只在两个同轮、同 rho 的更新都到达后才推进。

## 模块边界

- [parser.py](src/admm_consensus/parser.py)：解析并校验维度、矩阵对称正定性、rho、容差和资源限制。
- [kernel.py](src/admm_consensus/kernel.py)：本地二次子问题、共识更新、缩放对偶更新、原始/对偶残差和 rho 缩放。
- [coordinator.py](src/admm_consensus/coordinator.py)：固定顺序的状态机和事务式轮次推进。
- [repository.py](src/admm_consensus/repository.py)：原子 JSON 状态快照和 append-only JSONL 证据日志。
- [diagnostics.py](src/admm_consensus/diagnostics.py)：集中式参考解、任务载荷和状态序列化。
- [app.py](src/admm_consensus/app.py)：FastAPI 服务入口和 HTTP 错误契约。
- [worker.py](src/admm_consensus/worker.py)：单个本地计算进程。
- [local_runner.py](src/admm_consensus/local_runner.py)：启动两个 worker 子进程的便利驱动。

## ADMM 契约

缩放对偶形式为：

```text
x_i^{k+1} = argmin f_i(x_i) + y_i^k'(x_i-z^k) + rho^k/2 ||x_i-z^k||^2
z^{k+1}   = (x_1^{k+1}+x_2^{k+1})/2
y_i^{k+1} = y_i^k + rho^k (x_i^{k+1}-z^{k+1})
```

每轮严格按以下顺序记录证据：

1. 两个本地更新；
2. 共识变量 `z`；
3. 对偶变量 `y`；
4. 原始残差 `x_i-z` 和协调器对偶残差 `-rho(z_new-z_old)`；
5. 分别判断原始、对偶阈值；
6. 只有继续下一轮时，才按残差平衡规则调整 rho 并执行 `y_new_rho = y_old_rho * old_rho/new_rho`。

因此外部进程看到的 `round_number` 和 `rho` 必须同时匹配；早到、迟到、错轮、重复提交都不会污染本轮。若任一方超时，协调器丢弃已经收到的临时更新，轮次保持不变，可重新提交同一轮。

## 启动服务

系统 Python 已包含所需依赖，直接执行：

```bash
PYTHONPATH=src uvicorn admm_consensus.app:app --host 127.0.0.1 --port 8000
```

默认状态目录是 `./.admm_runs`。可通过环境变量调整本地教学限制：

```bash
ADMM_DATA_DIR=/tmp/admm-runs \
ADMM_MAX_DIMENSION=64 \
ADMM_MAX_ITERATIONS=1000 \
ADMM_MAX_RUNS=128 \
PYTHONPATH=src uvicorn admm_consensus.app:app
```

依赖版本锁定在 [requirements.lock](requirements.lock)。本工程没有要求创建虚拟环境，也不使用外部账号或业务数据。

## 自动轮次调用

服务端可以在单个 HTTP 请求里执行两个本地内核更新，适合检查数值行为：

```bash
RUN_ID=$(curl -sS -X POST http://127.0.0.1:8000/runs \
  -H 'content-type: application/json' \
  --data-binary @examples/asymmetric_problem.json \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["run"]["run_id"])')

curl -sS -X POST "http://127.0.0.1:8000/runs/$RUN_ID/auto-round" | python3 -m json.tool
```

也可以运行：

```bash
./examples/auto_round.sh
```

脚本会逐轮打印原始残差、对偶残差和 rho，并把事件日志写到 `examples/run-...-events.json`。

## 两个本地计算进程

先创建运行，然后开两个终端：

```bash
PYTHONPATH=src python3 -m admm_consensus.worker http://127.0.0.1:8000 "$RUN_ID" 1
PYTHONPATH=src python3 -m admm_consensus.worker http://127.0.0.1:8000 "$RUN_ID" 2
```

当两个 worker 都提交后，调用：

```bash
curl -sS -X POST "http://127.0.0.1:8000/runs/$RUN_ID/complete" | python3 -m json.tool
```

查看本轮任务：

```bash
curl -sS "http://127.0.0.1:8000/runs/$RUN_ID/task" | python3 -m json.tool
```

若 worker 2 超时：

```bash
curl -sS -X POST "http://127.0.0.1:8000/runs/$RUN_ID/timeout" \
  -H 'content-type: application/json' \
  -d '{"missing_worker_ids":[2],"reason":"deadline_exceeded"}'
```

响应中的 `round_number` 仍是当前轮，`pending_worker_ids` 为空，事件日志会记录丢弃的临时 worker。

## HTTP 错误类别

所有预期错误返回：

```json
{"error": {"code": "state_conflict", "message": "...", "details": {}}}
```

类别包括：

- `input`：矩阵形状、非有限数、非对称/非正定、非法 rho 或容差。
- `state_conflict`：错轮、错 rho、重复提交、终态提交、只有一方提交却要求完成。
- `resource_exhausted`：维度、最大轮数或运行目录数超过本地限制。
- `computation`：本地 KKT 系统求解失败、非有限结果或驻留条件不满足。
- `not_found`：运行或日志不存在。

## 证据与重放

每个 run 使用一个目录：

```text
.admm_runs/runs/<run_id>/
  state.json       # 最新原子快照，包括半轮 pending、z、y、rho、轨迹、参考解
  events.jsonl     # 每次提交、拒绝、完成、rho 调整、超时的追加证据
```

事件包含 run id、event id、UTC 时间、轮次、rho、阶段和关键中间状态。`state.json` 中的输入问题和配置足以重建求解器；事件日志足以解释每轮为何继续、收敛、达到最大轮数或丢弃整轮。

状态枚举为：

- `running`：等待或执行当前轮；
- `converged`：原始残差和对偶残差都满足各自阈值；
- `max_iterations`：达到最大轮数但至少一个残差未满足；
- `failed`：保留给不可恢复计算失败。

## 测试

执行：

```bash
python3 -m pytest -q
```

测试包含：

- 不对称规模的两个最小二乘局部模型；
- 用测试独立生成的 `A'b`、`A'A` 和 SciPy 正定型求解对照集中式最优解；
- 固定更新顺序和中断恢复；
- rho 改变前后的缩放对偶断言；
- 单进程超时、整轮丢弃和重放证据；
- 未达到容差时的 `max_iterations` 状态与终止理由；
- `input/state_conflict/resource_exhausted/computation/not_found` 分类；
- 两个真实本地 worker 子进程通过 HTTP 与 uvicorn 协同收敛。

## 剩余限制

- 这是本地教学后端，只实现契约要求的两个 worker；没有分布式注册中心、认证或横向扩缩容。
- 存储使用本地 JSON/JSONL，适合小维度和可教学的轨迹检查，不是高性能数据库。
- 超时由显式 API 调用表达，方便测试和重放；服务端不使用墙钟定时器强杀外部 worker。
- 自适应 rho 使用显式 `change_rounds` 的残差平衡规则，避免不可预测的隐式调度使轮次契约变模糊。
