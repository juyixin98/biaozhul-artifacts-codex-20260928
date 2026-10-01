# 两路不规则采样序列的时间滞后相关分析

本工程使用 Python、FastAPI、NumPy、SciPy、SQLite 实现两路采样时间不规则序列的滞后相关分析。运行所需依赖均在 `requirements.txt` 声明；当前机器已全局安装这些包，按约定直接使用 `python3`，不创建虚拟环境。

## 模块职责

- `lag_correlation/contracts.py`：HTTP 请求/响应契约、字段校验和错误入口。
- `lag_correlation/kernel/matching.py`：时间容忍窗与确定性一对多匹配。
- `lag_correlation/kernel/statistics.py`：线性去趋势、样本 z-score、Pearson 相关。
- `lag_correlation/kernel/analysis.py`：滞后网格、固定统计配置、选峰、失败类别和诊断事件。
- `lag_correlation/repository.py`：SQLite 版本结果与诊断持久化，不保存原始序列值。
- `lag_correlation/service.py`：契约模型到统计内核的应用服务映射。
- `lag_correlation/app.py`、`lag_correlation/__main__.py`：FastAPI 入口。
- `experiments/reference.py`：硬编码合成输入、具体期望和独立参考公式。
- `experiments/reproduce.py`：复现实验命令，逐场景输出检查项。
- `scripts/demo.py`：本地端到端演示，调用服务层并写入演示 SQLite。
- `tests/`：独立 pytest 套件，覆盖算法、契约、持久化和 HTTP 语义。

## 匹配规则

规则固定并在响应的 `matching_rule` 中回显：

1. 按时间间隔中位数选择较稀疏的一侧作为锚点侧；中位数相等时选择 `x`。
2. 对滞后 `lag`，锚点时间 `t_a` 寻找密集侧目标时间 `t_a - lag`。
3. `tolerance > 0`，匹配条件为 `abs(t_dense - (t_a - lag)) <= tolerance`。
4. 每个密集样本只分配给距离最近的一个锚点；等距时选择时间更早的锚点。
5. 同一锚点可接收多个密集样本，这些样本取值取均值，只产生一个配对。
6. 每个锚点最多出现一次，每个密集样本也最多使用一次。密集侧不会因为采样更密而在相关值中被重复加权。

配对明细 `pair_details` 包含锚点索引、密集样本索引、锚点时间、密集样本均值时间、时间差和滞后，可用于审计。

## 固定统计配置

- 去趋势：对每个滞后实际匹配到的两组值，分别按锚点时间做线性最小二乘去趋势。
- 标准化：样本 z-score，样本标准差使用 `ddof=1`。
- 有效配对下限：`6` 个“每锚点一个”的配对。
- 相关值：样本 Pearson 相关，限制到 `[-1, 1]`，按 12 位小数精度回显。
- 选峰：在接受的滞后中取最大相关；相关相同则优先绝对滞后更小，再优先更小的滞后值。

相关峰只是相关性最强的时间平移，不代表一方导致另一方。响应中的 `causality_notice` 固定说明这一点。

## 错误和未知语义

统计上无法判定与请求错误分开处理：

- HTTP 200 且 `status="unknown"`：请求合法，但没有可解释的相关峰。
  - `INSUFFICIENT_OVERLAP`：所有滞后的有效锚点配对数都少于 6，包括完全无重叠。
  - `INSUFFICIENT_VARIANCE`：至少一个达到配对数要求的滞后在去趋势后有一侧为常数，无法标准化。
- HTTP 422 `VALIDATION_ERROR`：时间不严格递增、长度不一致、非有限数、滞后网格非法等。
- HTTP 409 `DUPLICATE_REQUEST_ID`：该请求标识已有结果。
- HTTP 404 `RESULT_NOT_FOUND`：查询了不存在的请求标识。

诊断事件带 `request_id`、状态、配对数、密集样本使用/未匹配数、相关值和接受/拒绝原因。数据库和日志不写入原始样本值；日志不打印请求体，路径也做截断显示。

## 本地运行

运行测试：

```bash
python3 -m pytest -q
```

运行复现实验：

```bash
python3 -m experiments.reproduce
```

运行本地演示：

```bash
python3 scripts/demo.py
```

启动服务：

```bash
python3 -m lag_correlation
```

服务默认监听 `http://127.0.0.1:8000`。健康检查：

```bash
curl http://127.0.0.1:8000/health
```

可通过环境变量改变 SQLite 路径：

```bash
LAG_CORRELATION_DB_PATH=/path/to/result.sqlite3 python3 -m lag_correlation
```

## HTTP 契约示例

```json
{
  "request_id": "example-request",
  "x": {
    "name": "sparse_x",
    "times": [0.0, 10.0, 20.0],
    "values": [1.0, 4.0, 2.0]
  },
  "y": {
    "name": "dense_y",
    "times": [-2.1, -2.0, -1.9],
    "values": [0.9, 1.0, 1.1]
  },
  "lag_grid": {"min_lag": -5.0, "max_lag": 5.0, "step": 1.0},
  "matching": {"tolerance": 0.25}
}
```

提交和查询：

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/lag-correlation/analyses \
  -H 'Content-Type: application/json' \
  -d @request.json

curl http://127.0.0.1:8000/api/v1/lag-correlation/analyses/example-request
```

为了满足有效配对下限，真实请求每侧至少应提供足够覆盖 6 个锚点配对的样本；示例片段只展示字段形状。

## 复现实验内容

`experiments/reference.py` 中固定了 5 个合成场景：

- 已知偏移、不同采样密度：30 个密集样本聚合成 10 个锚点配对。
- 均匀采样特例：滞后 2 时 10 个点形成 8 个有效配对，并与独立 Pearson 参考值比较。
- 已知偏移加长缺口：缺口保留但整体仍可在正确滞后重合，10 个配对全部成立。
- 常数序列：去趋势后方差不足，返回 `unknown / INSUFFICIENT_VARIANCE`。
- 无有效重叠：返回 `unknown / INSUFFICIENT_OVERLAP`。

参考相关值使用实验模块中的独立 `numpy.polyfit` 和样本 Pearson 公式计算，测试期望不是由生产内核生成后再回放。
