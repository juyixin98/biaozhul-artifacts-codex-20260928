# 带参数上下界的非线性最小二乘信赖域服务

这是一个完全本地运行的 FastAPI 服务，用带显式参数尺度归一的信赖域方法求解

```text
min_theta  0.5 * sum_i (f(x_i, theta) - y_i)^2
s.t.       lower <= theta <= upper
```

模型 `f` 来自受限表达式：解析 Python AST，只允许数值常数、参数、变量 `x`、四则运算、幂、括号和白名单数学函数 `exp/log/sqrt/sin/cos/tan/log10/abs`。导数由反向自动微分生成，并在求解前用独立有限差分复核。

## 工程结构

- `app/expression.py`：受限表达式、安全校验、反向 AD 和非法函数值异常。
- `app/solver.py`：Gauss-Newton 信赖域内核、缩放坐标、主动集 Steihaug-CG、边界投影梯度和状态分类。
- `app/diagnostics.py`：迭代证据、秩/奇异值/条件数、AD 差分误差、版本信息和 JSONL 日志。
- `app/schemas.py`：Pydantic 请求与响应结构。
- `app/main.py`：FastAPI 路由和错误映射。
- `app/config.py`：日志路径等环境配置。
- `tests/`：表达式、求解器和 HTTP 层独立测试。
- `scripts/demo.py`：无需启动服务的本地合成数据演示。

## 信赖域方法

每个参数使用用户提供的正尺度 `scale`；未提供时使用

```text
scale_j = max(|initial_j|, upper_j-lower_j, 1)
```

优化变量使用缩放坐标 `theta = S z`。因此一个单位的信赖域半径对不同数量级参数具有可比含义。局部模型为

```text
m(q) = g_z^T q + 0.5 q^T (J S)^T (J S) q
```

子问题同时受信赖域球和缩放后的参数盒约束，使用主动集 Steihaug-CG 求候选步。每轮记录：

- 预测下降 `predicted_reduction`
- 实际下降 `actual_reduction`
- 下降比 `rho = actual / predicted`
- 信赖域半径、步长、接受/拒绝、试算次数
- 当前触及的上下界参数
- 内点驻点和边界驻点布尔判定

一阶停止准则使用缩放投影梯度。下界处仅当梯度非负才阻挡下降；上界处仅当梯度非正才阻挡下降。因此：

- `CONVERGED_INTERIOR`：没有有效活动界，投影梯度满足容差。
- `CONVERGED_BOUNDARY`：至少一个参数位于有效活动界上，且投影梯度满足容差。

## 错误与状态语义

求解层失败不会被统一返回成成功：

- `CONVERGED_INTERIOR` / `CONVERGED_BOUNDARY`：成功，`success=true`。
- `MAX_ITERATIONS`：达到最大迭代次数仍未满足驻点条件。
- `NON_CONVERGED`：信赖域收缩、无数值有效步等原因导致不能继续。
- `UNIDENTIFIABLE`：缩放雅可比列秩不足；响应包含秩、奇异值和条件数。
- `ILLEGAL_FUNCTION_VALUE`：模型或导数在初值/候选点产生 NaN、无穷或越界值。
- `DERIVATIVE_CHECK_FAILED`：AD 雅可比与有限差分校验误差超过阈值。
- HTTP `400`：表达式不合法等请求级模型错误。
- HTTP `422`：Pydantic 字段、上下界或数据长度校验失败。

正常 HTTP `200` 也可能带 `success=false`，因为数值求解失败是一种明确计算结果，不是服务崩溃。

## 安装与运行

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

求解请求示例：

```bash
curl -s http://127.0.0.1:8000/solve \
  -H 'Content-Type: application/json' \
  -d '{
    "expression": "amp*exp(-rate*x)+floor",
    "run_id": "curl-example",
    "parameters": {
      "amp": {"initial": 0.1, "lower": 0, "upper": 10, "scale": 2},
      "rate": {"initial": 0.01, "lower": 0, "upper": 10, "scale": 1.5},
      "floor": {"initial": 1.5, "lower": 0, "upper": 10, "scale": 1}
    },
    "x": [0, 0.5, 1, 1.5, 2],
    "y": [2.3, 1.249, 0.745, 0.511, 0.401],
    "options": {"max_iterations": 80, "tolerance": 0.0001}
  }'
```

OpenAPI 文档位于 `http://127.0.0.1:8000/docs`。

## 本地演示

```bash
python3 scripts/demo.py
```

演示包含：

- 合成指数衰减模型 `amp*exp(-rate*x)+floor`，使用远离参考解的坏初值。
- 合成趋近饱和模型 `a*(1-exp(-k*x))/(1+b*(1-exp(-k*x)))`。
- 强相关参数的高条件数诊断。
- 非法 `sqrt` 定义域和参数冗余不可辨识的独立失败类别。
- 参考解、估计参数误差、残差 RMS，以及每轮预测/实际下降。

## 测试复现

```bash
pytest -q
```

测试不是“接口能调用”检查：

- 参考解由独立 NumPy 公式硬编码生成，不由被测核心产生。
- 断言参数误差、残差、条件数、秩、AD 差分误差和内点/边界类别。
- 对最大迭代、不可辨识、非法函数值分别断言具体状态。
- 检查 JSONL 日志的 `run_id` 关联、版本、求解开始、差分校验、逐轮步和最终判定。

## 日志

默认日志文件为 `logs/runs.jsonl`，可通过环境变量修改：

```bash
NLS_LOG_PATH=/tmp/nls.jsonl uvicorn app.main:app --host 127.0.0.1 --port 8000
```

每行是一个 JSON 对象，均带 `run_id` 和版本信息。请求可以显式传入 `run_id`，便于把输入、服务日志和测试结果关联起来。
