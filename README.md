# 有界整数线性规划后端

这是一个基于 FastAPI 的小型 MILP 后端。模型解析、分支定界内核、候选解诊断、运行日志和 HTTP 入口拆分为独立模块；每个节点的线性规划松弛由 SciPy 的 HiGHS 求解器处理。

实现只把 HiGHS 当作线性规划松弛求解器，没有调用它的内置 MIP 分支定界；整数搜索、上下界、剪枝和预算逻辑都在本项目内核中完成。

## 模块结构

- `milp/schemas.py`：Pydantic 请求、响应和事件结构。
- `milp/model.py`：变量、目标和约束解析；最大化统一转换成内部最小化，`>=` 转成标准上界约束。
- `milp/solver.py`：分支定界、上下界维护、按目标方向剪枝、时间/节点预算和可恢复开放节点。
- `milp/diagnostics.py`：不使用求解器结论，独立检查整数残差、变量边界、不等式和等式残差。
- `milp/journal.py`：每次运行写入可重放 JSONL，包含原始问题、选项、关键中间状态、判断理由和最终结果。
- `milp/app.py`：FastAPI 服务入口和错误类别映射。
- `tests/brute_force.py`：小范围整数变量的独立穷举参考实现，不调用被测内核。

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock
```

当前环境锁定版本：

- Python 3.12
- FastAPI 0.141.1
- Uvicorn 0.54.0
- NumPy 2.5.3
- SciPy 1.18.1
- Pydantic 2.13.5
- pytest 9.1.1
- httpx 0.28.1

## 启动服务

```bash
uvicorn milp.app:app --host 127.0.0.1 --port 8000
```

健康检查：

```bash
curl -sS http://127.0.0.1:8000/health
```

示例请求：

```bash
curl -sS http://127.0.0.1:8000/api/solve \
  -H 'content-type: application/json' \
  --data-binary @examples/knapsack.json
```

也可以直接运行：

```bash
bash examples/call_api.sh
```

## 请求格式

`POST /api/solve` 接收：

```json
{
  "problem": {
    "name": "knapsack",
    "sense": "maximize",
    "objective": {"x1": 5, "x2": 3},
    "variables": [
      {"name": "x1", "type": "binary"},
      {"name": "x2", "type": "integer", "lower": 0, "upper": 4},
      {"name": "y", "type": "continuous", "lower": 0, "upper": 10}
    ],
    "constraints": [
      {"name": "c1", "terms": {"x1": 1, "y": 2}, "sense": "<=", "rhs": 8}
    ]
  },
  "options": {
    "run_id": "stable-id",
    "integer_tolerance": 1e-8,
    "constraint_tolerance": 1e-8,
    "lp_tolerance": 1e-10,
    "time_limit_seconds": 5,
    "node_limit": 100
  }
}
```

约束方向支持 `<=`、`>=`、`==`。整数和二进制变量必须给出有限边界；这避免把无界整数松弛误报为可解。

## 状态和缺口

响应中的主要状态：

- `optimal`：所有开放节点均已处理或按界剪枝，`absolute_gap` 为 0。
- `infeasible`：未找到可行整数解，所有节点均不可行或被剪枝。
- `resource_exhausted`：时间或节点预算到期；响应给当前 incumbent、`best_bound`、绝对/相对缺口和开放节点，绝不标成 optimal。
- `unbounded`：纯连续 LP 松弛无界。
- `relaxation_unbounded`：含整数变量的某个 LP 松弛无界；由于无界松弛不一定提供可分支的整数点，服务停止并给出已有 incumbent，不宣称整数最优或无界。

内部始终最小化。若原问题是最大化，目标系数取负；返回时目标值和边界再换回原最大化方向。因此剪枝条件始终为：松弛下界不小于当前最小化 incumbent 时剪枝。

整数候选只有在满足整数容差后才四舍五入，随后必须通过独立的边界和约束复核。容差只用于决定“看起来是否整数”，不会让四舍五入后的非法解直接被接受。

## JSONL 日志

默认写入 `logs/<run_id>.jsonl`。三类记录分别是：

- `run_start`：原始问题与选项，可直接重放。
- `event`：LP 结果、分支变量、分支值、子节点约束、节点界、incumbent、剪枝原因和诊断残差。
- `run_complete`：最终状态、目标值、界、缺口和节点数量。

非有限浮点在严格 JSON 中序列化为 `"Infinity"`、`"-Infinity"` 或 `"NaN"`。重复 `run_id` 返回 `state_conflict`，防止覆盖审计日志。

## 配置

环境变量：

- `MILP_JOURNAL_DIR`，默认 `logs`
- `MILP_INTEGER_TOLERANCE`，默认 `1e-8`
- `MILP_CONSTRAINT_TOLERANCE`，默认 `1e-8`
- `MILP_LP_TOLERANCE`，默认 `1e-10`
- `MILP_DEFAULT_TIME_LIMIT_SECONDS`，默认 `30`
- `MILP_DEFAULT_NODE_LIMIT`，默认 `10000`
- `MILP_MAX_EVENTS_IN_RESPONSE`，默认 `200`；日志仍保留全部事件

## 测试

```bash
pytest -q
```

测试包含：

- 与独立穷举 oracle 对照的 0-1 背包最优值。
- 根不可行、松弛可行整数不可行、整数容差候选仍因约束被拒绝。
- 节点预算和时间预算停止，检查 incumbent、原方向最优界和缺口。
- 每个分支事件和剪枝原因。
- 开放节点恢复所需的分支约束和当前界。
- 输入错误、状态冲突、资源耗尽和计算失败的分类。
- FastAPI 请求、响应和真实 JSONL 日志。

## 剩余限制

- 当前节点选择是深度优先并结合最小松弛界，适合教学和中小规模问题；未实现热启动、割平面、分支规则打分或并行搜索。
- 穷举测试 oracle 只用于小整数域，不应作为大规模算法。
- LP 时间限制由 HiGHS 选项传入；分支定界外层还会检查总时间预算。若 HiGHS 在极端数值情况下返回数值困难，接口归类为 `computation_failed`，而不是伪造最优解。
- 日志按每次运行一个 JSONL 文件持久化，尚未实现服务端分页查询或压缩归档。
