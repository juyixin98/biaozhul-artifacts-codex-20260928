# 精确置换检验服务

这是一个本地运行的后端项目，用 FastAPI、NumPy、SciPy 和 SQLite 实现两组独立样本均值差的双侧置换检验。项目只使用本地合成输入和本地 SQLite，不依赖生产账号或外部业务服务。

## 工程结构

```text
permutation_service/
  core.py       # 统计内核：精确枚举、Monte Carlo、取消点、参数校验
  service.py    # 后台作业、输入哈希、进度事件、错误持久化
  storage.py    # SQLite 表、运行状态、审计事件
  api.py        # FastAPI 路由、HTTP 状态码、错误映射
  schemas.py    # 请求/响应契约
  config.py     # 环境变量配置
  versions.py   # Python、NumPy、SciPy、FastAPI、SQLite 版本
tests/          # 独立统计、服务、存储测试
scripts/demo_local.py
```

## 统计定义

输入为合并后的观测值 `values`，前 `group1_size` 个值属于第一组，其余属于第二组。检验统计量为：

```text
T = abs(mean(group1) - mean(group2))
```

所有重新分组都保留两组大小。精确模式枚举样本身份组合 `C(n, group1_size)`，所以两个数值相同但样本身份不同的观测仍按不同排列计数。双侧 p 值使用 `#(T_perm >= T_observed) / C(n, group1_size)`，等号计入极端。

当组合数不大于 `exact_max_combinations` 时，`auto` 使用精确枚举；否则使用 Monte Carlo。Monte Carlo 用传入或自动生成的整数种子创建 NumPy `default_rng(seed)`，抽样时保留组大小并剔除“原始标签”这一身份排列。有限样本 p 值固定为加一估计：

```text
p = (#extreme_sampled + 1) / (B + 1)
SE = sqrt(p * (1 - p) / (B + 1))
95% interval = p ± 1.96 * SE，裁剪到 [0, 1]
```

结果中的 `evaluated_permutations` 是实际评估的有效排列数；`total_possible_assignments` 是总身份组合数；`extreme_permutations` 是按加一修正后的极端数。

## 启动服务

系统 Python 已有所需依赖，无需创建虚拟环境：

```bash
python3 -m uvicorn permutation_service.api:app --host 127.0.0.1 --port 8000
```

也可以运行：

```bash
python3 -m permutation_service
```

健康检查和版本：

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/version
```

创建检验：

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/runs \
  -H 'content-type: application/json' \
  -d '{"values":[1,2,3],"group1_size":1}'
```

返回中的 `run_id` 可用于查询状态和事件：

```bash
curl -s http://127.0.0.1:8000/api/v1/runs/<run_id>
curl -s http://127.0.0.1:8000/api/v1/runs/<run_id>/events
curl -s -X POST http://127.0.0.1:8000/api/v1/runs/<run_id>/cancel
```

## 请求字段

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `values` | 必填 | 合并观测值，必须是有限一维数值列表 |
| `group1_size` | 必填 | 第一组样本数，范围 `[1, n-1]` |
| `alpha` | `0.05` | 显著性阈值，要求 `0 < alpha < 1` |
| `method` | `auto` | `auto`、`exact`、`monte_carlo` |
| `monte_carlo_replications` | `9999` | Monte Carlo 有效抽样数 `B` |
| `exact_max_combinations` | `100000` | `auto` 精确枚举阈值；`0` 强制 Monte Carlo |
| `random_seed` | 自动 | 非负整数；相同输入和种子可复现 RNG 结果 |
| `return_null_distribution` | `false` | 内核是否返回完整零分布 |

显式 `method="exact"` 会执行精确枚举，但受安全上限 `PERM_REQUESTED_EXACT_LIMIT_CAP` 限制，默认 100,000,000。超过上限会同步返回 `EXACT_LIMIT_EXCEEDED`。

## HTTP 和错误语义

`POST /api/v1/runs` 成功接收时返回 `202 Accepted`。查询运行时：

- `200`：完成，`result` 中有统计结果。
- `202`：仍在 `pending` 或 `running`，没有显著性结论。
- `400`：可预知输入或请求问题，如 `INVALID_INPUT`、`INVALID_REQUEST`、`EXACT_LIMIT_EXCEEDED`。
- `404`：运行不存在，`RUN_NOT_FOUND`。
- `409`：运行已取消，错误码为 `CANCELED`；或终态冲突。
- `500`：未知计算或服务异常，错误码为 `UNKNOWN_ERROR`。

取消是独立终态：状态为 `canceled`，`result` 必须为 `null`，不能返回 `significant=false` 或任何 p 值。服务重启前仍处于 `pending/running` 的任务会在启动时标记为 `failed / SERVICE_INTERRUPTED`，不会伪装为成功或不显著。

每个运行都有稳定 `run_id` 和规范化输入的 SHA-256 `input_hash`。事件表记录接收、依赖版本、进度批次、取消、最终 p 值和判定依据。

## 本地演示

```bash
python3 scripts/demo_local.py
```

演示使用 FastAPI 的本地 `TestClient` 和 `data/demo.sqlite`，包含一个可手算的精确用例、一个带种子的 Monte Carlo 用例、一个同步输入错误和审计事件输出。

## 配置

可通过环境变量调整：

```text
PERM_DB_PATH                              默认 data/runs.sqlite
PERM_DEFAULT_MONTE_CARLO_REPLICATIONS     默认 9999
PERM_EXACT_MAX_COMBINATIONS               默认 100000
PERM_REQUESTED_EXACT_LIMIT_CAP            默认 100000000
PERM_MAX_VALUES                           默认 10000
PERM_MAX_MONTE_CARLO_REPLICATIONS         默认 10000000
PERM_LOG_LEVEL                            默认 INFO
```

## 测试和复现

运行全部测试：

```bash
python3 -m pytest -q
```

测试不是只调用接口。它们断言：

- `[1, 2, 3]`、第一组大小 1 的完整排列分布为 `(0, 1.5, 1.5)`，p 值 `2/3`。
- 测试文件内独立的 `itertools.combinations` 参考实现逐点比较内核零分布。
- 标签交换前后统计量和 p 值一致。
- 全同值、重复数值、极端不平衡组的具体 p 值和计数。
- 固定 Monte Carlo 种子重放出相同零分布。
- 抽样标签每组始终为指定大小。
- 加一修正在唯一极端观测下给出 `1/(B+1)`，并报告标准误和区间。
- 精确枚举和服务层取消都成为显式 `CANCELED`，不产生显著性结论。
- schema、内核和未知错误使用独立错误码，未知状态不会统一返回成功。

依赖清单见 `requirements.txt`，项目元数据见 `pyproject.toml`。

