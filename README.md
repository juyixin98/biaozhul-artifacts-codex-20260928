# 受限 STL 式季节趋势分解后端

本项目实现一个可通过 FastAPI 调用的受限 STL（Seasonal-Trend decomposition
using LOESS）后端。算法使用本地 LOESS 平滑、固定边界外推和双平方鲁棒权重迭代。

## 固定统计契约

参数不由请求传入：

| 项 | 固定值 |
| --- | --- |
| 周期 `period` | 7 |
| 季节子序列窗口 `seasonal_window` | 7 |
| 趋势窗口 `trend_window` | 15 |
| 低通 LOESS 窗口 `low_pass_window` | 9 |
| LOESS 阶数 | 全部为 1 |
| 内迭代次数 | 2 |
| 鲁棒外迭代次数 | 15 |
| jump | 1，不做跳点插值近似 |

季节平滑对每个相位子序列分别做局部线性 LOESS，并在首尾各外推一个周期。
扩展后的季节序列依次经过周期长度移动平均、周期长度移动平均、三点移动平均和
低通 LOESS，从而去掉季节平滑中的低频部分。最终季节项减去样本均值，均值常数
平移到趋势项，以保证季节项中心化、分解位置可辨识，并保持加法重构：

```text
observed = trend + seasonal + residual
mean(seasonal) = 0
```

鲁棒权重使用 STL 双平方权重，尺度为基于残差绝对值中位数的 `6 * MAD`。
残差大于约 `0.999 * 6 * MAD` 的样本权重为 0。

## 输入边界

- 输入必须是一维有限数值序列。
- `NaN`、`Infinity` 和缺测值直接拒绝，不会隐式填零或前向填充。
- 最少 14 个观测，即至少两个完整周期。少于 14 个观测返回
  `input_contract_violation`。
- 默认单次最多 100,000 个观测；超过限制返回 `resource_exhausted`。
- `run_id` 可作为幂等键。重复使用同一个 `run_id` 返回 `state_conflict`，
  不会覆盖已有运行。

## 模块结构

- `stl_backend/kernel.py`：样本校验、LOESS、季节扩展、低通、鲁棒权重和 STL 内核。
- `stl_backend/service.py`：应用服务、结果诊断、请求指纹和事务编排。
- `stl_backend/repository.py`：SQLite 表结构、运行记录和迭代事件。
- `stl_backend/api.py`：FastAPI 路由与错误到 HTTP 状态码的映射。
- `stl_backend/schemas.py`：Pydantic 请求和响应契约。
- `scripts/generate_fixtures.py`：生成确定性合成夹具和独立参考答案。
- `scripts/verify_decomposition.py`：复现实验和验收报告。
- `scripts/replay_run.py`：读取 SQLite 中的运行号、状态和判断理由。
- `tests/`：内核、服务、API 与边界行为测试。

## 合成夹具和独立答案

夹具位于 `tests/fixtures/robust_stl.json`，生成规则为：

- 140 个观测；
- 已知线性趋势 `2 + 0.08 * index`；
- 周期 7 的已知正弦季节项；
- 固定随机种子的高斯噪声；
- 3 个明确的大幅离群点，位置为 24、71、117。

参考答案由 `statsmodels==0.15.0` 的 `statsmodels.tsa.seasonal.STL`
成熟实现生成，不由本项目被测内核生成。夹具脚本只应在需要刷新独立参考时运行：

```bash
python3 scripts/generate_fixtures.py
```

## 运行 API

```bash
python3 -m uvicorn stl_backend.api:app --host 127.0.0.1 --port 8000
```

示例：

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/decompositions \
  -H 'Content-Type: application/json' \
  -d '{"run_id":"demo-001","values":[...至少14个有限数字...]}'
```

其他端点：

- `GET /health`：服务健康状态和内核版本。
- `GET /api/v1/config`：固定参数、样本限制和版本。
- `GET /api/v1/runs/{run_id}`：运行状态、指纹、错误或完整结果。
- `GET /api/v1/runs/{run_id}/events`：契约判断、每次外迭代关键状态和验收判断。

数据库路径默认是当前目录下的 `stl_runs.sqlite3`，可用环境变量
`STL_BACKEND_DB` 覆盖。

## 运行日志和重放

SQLite 中有两张表：

- `runs`：运行号、运行 ID、状态、观测数、SHA-256 请求指纹、配置、结果和错误。
- `run_events`：运行号、迭代编号、阶段、判断理由和 JSON 中间状态。

每次成功分解记录契约接收事件、16 个外迭代状态（第 0 次为均匀权重初始化，
第 1 至 15 次应用鲁棒权重）和最终验收事件。中间状态包括最大残差、均方根残差、
下一轮权重最小值、零权重数、MAD 截止值和加法重构误差。

重放：

```bash
python3 scripts/replay_run.py demo-001 --database stl_runs.sqlite3
```

## 错误契约

| code | HTTP | 含义 |
| --- | --- | --- |
| `input_contract_violation` | 400 | 非数值、非一维、缺测、无穷值、少于两个周期 |
| `state_conflict` | 409 | 重复 `run_id` 或状态冲突 |
| `resource_exhausted` | 507 | 观测数超限或 SQLite/存储资源失败 |
| `calculation_failed` | 422 | 数值计算失败或产生非有限结果 |
| `run_not_found` | 404 | 查询未知运行 ID |

## 验证

安装运行时与测试依赖：

```bash
pip install -r requirements-test.txt
```

在本机已预装依赖的环境中可直接运行：

```bash
python3 -m pytest -q
python3 scripts/verify_decomposition.py
```

验证脚本会输出并写入 `reports/verification.json`，包含：

- 固定窗口和周期断言；
- 季节项中心化断言；
- 加法重构断言；
- 已知离群点权重为 0 的断言；
- 趋势、季节、残差、鲁棒权重与 `statsmodels` 参考的最大绝对差；
- 所有组件有限性检查。

本次开发中实际执行并通过：

```text
22 passed
```

未执行项：无。
