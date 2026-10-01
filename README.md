# 带上下界的非线性最小二乘服务

这是一个本地可运行的 FastAPI 工程，用显式参数缩放的信赖域方法求解

```text
min_theta  0.5 * sum_i w_i^2 * (f(x_i, theta) - y_i)^2
s.t.       lower <= theta <= upper
```

模型来自受限 Python 表达式。表达式经过 AST 白名单解析，导数由前向对偶数自动微分得到，并在求解前使用中心或单边有限差分复核。所有数据均为本地合成数据或请求中提供的数据，不连接外部业务系统。

## 工程结构

```text
app/core/dual.py           # 前向对偶数自动微分
app/core/expression.py     # 受限表达式、模型残差、有限差分复核
app/core/solver.py         # 有界、显式缩放信赖域内核
app/core/diagnostics.py    # 状态、失败类别、迭代和可辨识性诊断
app/schemas.py             # API 请求/响应模型
app/service.py             # 编排、输入指纹、结构化运行日志
app/main.py                # FastAPI 入口和错误映射
tests/                     # 内核、表达式、API、日志测试
scripts/demo.py            # 本地合成指数模型演示
```

## 快速开始

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

本地演示不依赖真实网络服务，使用 FastAPI `TestClient` 调用同一套应用路由：

```bash
python3 scripts/demo.py
```

运行测试：

```bash
pytest
```

## 请求示例

```bash
curl -sS http://127.0.0.1:8000/api/v1/least-squares/fit \
  -H 'Content-Type: application/json' \
  -d '{
    "expression": "a*exp(-b*x)+c",
    "parameter_names": ["a", "b", "c"],
    "x": [0, 0.5, 1, 1.5, 2],
    "y": [3.4, 2.198, 1.468, 1.025, 0.757],
    "initial": [0.2, 2.0, 0.01],
    "bounds": {
      "lower": [0, 0, 0],
      "upper": [10, 5, 10]
    },
    "scales": [3, 1, 0.5],
    "options": {
      "check_derivatives": true
    }
  }'
```

`bounds.lower` 和 `bounds.upper` 的元素可用 `null` 表示对应方向无界。`scales` 必须为正数；不提供时系统会根据初值和有限边界宽度生成显式归一化尺度。

## 支持的表达式

支持数值常量、参数名、变量 `x`、括号、一元正负号、`+ - * / **`，以及白名单函数：

```text
exp, log, sqrt, sin, cos, tan, tanh, abs
```

不允许属性访问、下标、名字调用、lambda、导入或任意 Python 表达式。变量名可以通过 `variable_name` 修改，但不能和参数重名。

## 数值方法

内部使用归一化坐标 `z = theta / scales` 求解自由变量的信赖域子问题：

```text
min  g_free^T p + 0.5 p^T J_free^T J_free p
s.t. ||p||_2 <= radius
```

子问题由对称特征值分解和一维参数搜索求解。物理边界通过 0.95 fraction-to-boundary 规则裁剪。每轮保存：

- 实际下降 `cost_old - cost_trial`
- 模型预测下降
- 二者比值 `rho`
- 接受/拒绝和信赖域半径
- 归一化步长、投影梯度范数
- 活动下界、活动上界、本轮碰到的边界
- 明确的判定依据

边界驻点通过投影一阶条件识别：

- 在下界且梯度非负时，下界活动。
- 在上界且梯度非正时，上界活动。
- 完全无活动参数为 `interior`。
- 部分参数活动为 `mixed`。
- 全部活动为 `boundary`。

## 状态和错误语义

求解结束或失败时不会把未知状态统一包装成成功。

| 状态 | HTTP | 含义 |
| --- | --- | --- |
| `OPTIMAL` | 200 | 投影一阶梯度满足容差 |
| `CONVERGED_COST_TOLERANCE` | 200 | 相对目标变化已足够小；仍是成功状态 |
| `CONVERGED_STEP_SIZE` | 200 | 归一化步长已足够小；仍是成功状态 |
| `NON_CONVERGED_MAX_ITERATIONS` | 200 | 到达迭代上限但未驻留，`success=false` |
| `NON_CONVERGED_TRUST_REGION` | 200 | 半径过小且无有效下降，`success=false` |
| `UNIDENTIFIABLE` | 200 | 到达驻留但缩放雅可比秩亏，`success=false` |
| `INVALID_TRIAL_VALUE` | 200 | 试探点函数值或导数非法次数超限，`success=false` |
| `INVALID_INITIAL_RESIDUAL` | 422 | 初值处残差、模型值或雅可比为 NaN/无穷/越域 |
| `DERIVATIVE_MISMATCH` | 422 | 自动微分未通过有限差分复核 |
| `INVALID_EXPRESSION` | 400 | 表达式语法非法、标识符未知或包含禁止结构 |
| `INVALID_INPUT` | 400 | 维度、边界、尺度、权重等请求输入不合法 |
| `INTERNAL_ERROR` | 500 | 未预期错误；日志记录但不会返回成功 |

可辨识性诊断对缩放雅可比 `J * diag(scales)` 做 SVD。秩亏时返回奇异值和数值秩；满秩时给出条件数和参数相关系数矩阵。

## 日志和可追溯性

日志默认输出 JSON。每次运行有：

- `run_id`：本次运行 UUID
- `input_fingerprint`：规范化请求的 SHA-256 前 16 位
- 服务、NumPy、SciPy 版本
- 进度，例如 `3/100`
- 每轮实际下降、预测下降、比值、半径、活动边界和判定依据
- 最终状态、原因、数值秩和迭代次数

设置 `NLS_LOG_JSON=0` 可切换为普通文本日志，设置 `NLS_LOG_LEVEL=DEBUG` 可提高日志详细度。

## 已保留的验证

测试中包含合成指数衰减 `a*exp(-b*x)+c` 与饱和模型 `A*x/(K+x)`，使用坏初值和强相关参数场景。参考数据由显式真值和固定随机种子生成，独立参考解由 SciPy `optimize.least_squares(method="trf")` 生成；被测内核不会生成这些参考答案。

测试断言具体参数、残差、相关系数、活动边界、每轮实际/预测下降，以及以下独立失败类别：

- 迭代上限非收敛
- 精确共线导致不可辨识
- 初值非法函数值
- 试探点非法函数值
- 自动微分与有限差分不一致
- 非法或危险表达式

当前执行结果：

```text
pytest
20 passed
```
