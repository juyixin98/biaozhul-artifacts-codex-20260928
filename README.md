# 带约束二次规划主动集后端

这是一个从空目录搭建的本地 FastAPI 后端，用于求解稠密凸二次规划：

```text
minimize    1/2 x'Hx + c'x
subject to  Gx <= h,  Ax = b
```

输入中的 `>=` 约束会在解析层翻号为统一的 `<=` 形式。所有数据、夹具和参考求解器都来自本地请求或本机依赖，不访问外部业务系统，也不使用真实账号数据。

## 技术栈与模块

- Python 3.11+，本地验证使用 Python 3.12.3。
- FastAPI 与 Pydantic：服务入口、请求模型和错误类别。
- NumPy：稠密矩阵、目标函数、残差和事件数据。
- SciPy：SVD、列选主元 QR、Cholesky/最小二乘，以及独立参考求解器。
- Pytest/HTTPX：独立单元和接口测试。

模块分工如下：

- `app/parser.py`：维度、有限值、Hessian 对称性和正定性校验，约束标准化。
- `app/core/feasibility.py`：第一阶段可行性 LP，独立调用 `scipy.optimize.linprog(method="highs")`。
- `app/core/linalg.py`：SVD 秩检测、列选主元 QR 基行选择、零空间 KKT 面求解。
- `app/core/active_set.py`：第二阶段原始主动集主迭代，处理阻塞约束加入和负乘子移除。
- `app/core/diagnostics.py`：最终点上的独立 KKT 证据，覆盖原始、对偶、稳定条件和互补条件。
- `app/service.py`：编排、状态归类、请求指纹、版本和处理位置。
- `app/observability.py`：带请求 ID 的 JSON 结构化日志。
- `app/main.py`：HTTP 接口和错误响应。
- `tests/`：不依赖被测核心生成参考值的测试夹具和断言。

## 关键取舍

可行性和主迭代严格分开。传入的 `initial_point` 如果满足原始约束，可以直接作为第二阶段起点；否则第一阶段调用 HiGHS 求解可行性 LP。不可行问题不会进入主动集主迭代。

主动集在每个工作面上求解切空间二次问题。系统先通过 SVD 找约束矩阵秩和零空间，再对正定缩约 Hessian 使用 Cholesky；秩亏时不组装或求逆奇异 KKT 矩阵。列选主元 QR 只负责选择代表性独立基行，依赖行会保留残差诊断但不盲目加入工作面。

阻塞约束按线搜索最小步长加入，并使用 Bland 风格的最小索引平局规则；数值零方向下只有乘子小于 `-dual_tol` 的不等式会被移除。最终 KKT 乘子不是简单复用中间迭代量，而是在最终点重新用非负最小二乘计算。

返回结果把硬性失败和不确定结论分开：

- `failure_reasons`：不可行、非正定 Hessian、维度错误或 KKT 残差超差等明确问题。
- `uncertainties`：迭代上限、近极限条件数、最终残差略超容差但仍可给出数值点等不确定结论。
- `events`：第一阶段、等秩检测、初始终活动集、加约束、删约束、线搜索步长和最终工作面。

## 安装与启动

推荐使用虚拟环境：

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.lock
```

启动服务：

```bash
python3 -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

OpenAPI 文档位于 `http://127.0.0.1:8000/docs`。

## 示例请求

求解边界最优示例：

```bash
curl -s http://127.0.0.1:8000/api/v1/qp/solve \
  -H 'Content-Type: application/json' \
  --data @examples/boundary_qp.json
```

预期最优点约为 `[0.0, 1.0]`，活动约束为 `sum_leq_1` 和 `x_nonnegative`，目标值为 `-3.0`。

不可行示例：

```bash
curl -s http://127.0.0.1:8000/api/v1/qp/solve \
  -H 'Content-Type: application/json' \
  --data @examples/infeasible_qp.json
```

只运行第一阶段可行性获取：

```bash
curl -s http://127.0.0.1:8000/api/v1/qp/feasibility \
  -H 'Content-Type: application/json' \
  --data @examples/feasibility_only.json
```

不开 HTTP 服务也可以运行：

```bash
python3 examples/run_examples.py
python3 scripts/verify_with_reference.py
```

## 容差

可在请求的 `tolerances` 中配置：

- `primal_tol`：等式和不等式原始可行性，默认 `1e-8`。
- `dual_tol`：稳定条件和负乘子判定，默认 `1e-8`。
- `complementarity_tol`：互补松弛，默认 `1e-8`。
- `active_tol`：判定约束是否在活动边界上，默认 `1e-9`。
- `rank_tol`：SVD/QR 秩容差，默认 `1e-11`。
- `step_tol`：数值零方向，默认 `1e-10`。
- `symmetry_tol`、`positive_definite_tol`：Hessian 输入校验。

## 验证案例

`tests/conftest.py` 中合成了以下案例：

- 边界最优：最优点位于两个线性约束交点。
- 冗余约束：完全重复、翻号重复和严格冗余不等式并存。
- 近退化：近线性相关但仍独立的约束在小容差下被识别并加入/移出。
- 秩亏等式：重复等式只保留独立基行。
- 负乘子删除：显式传入角落初始点，验证活动约束被移除。
- 无可行解：由 HiGHS 独立判定为不可行。

可行问题的参考解由 SciPy SLSQP 独立优化得到；不可行问题由 SciPy HiGHS 独立判定。测试不只检查接口状态，还断言最优点、活动集、KKT 残差、事件类型和失败类别。

运行测试：

```bash
pytest -q
```

## 支持范围

当前支持小规模、稠密、严格正定的 QP。适合教学、算法验证、本地合成数据和服务原型。暂不支持非凸 Hessian、半正定目标、整数变量、稀疏大规模输入、外部数据源以及生产级热启动工作队列。

如果问题极度病态，服务会保留数值点和诊断信息，但可能返回 `uncertain`，而不是把未通过 KKT 的结果宣称为最优。
