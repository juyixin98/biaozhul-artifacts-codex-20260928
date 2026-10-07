# PAV 概率校准后端

这是一个本地 Python 多模块后端，用加权 PAV（Pool-Adjacent-Violators）把原始分数校准为非递减的二分类概率。项目使用 FastAPI、NumPy、SQLite，测试使用 pytest、httpx 和 SciPy，不依赖外部业务服务。

## 算法契约

- 输入为有限实值 `score`、二分类整数 `target`（0 或 1）和非负 `weight`。
- 缺失或显式为 `null` 的权重按 1.0 处理；零权重记录不参与训练，并在响应 `warnings` 中列出。
- 缺失 `score` 或 `target`、非有限数值、非二分类目标、负权均为失败，响应在 `failures` 中给出具体类别。
- 相同分数先按权重聚合：平均目标为 `sum(w*y)/sum(w)`，权重为 `sum(w)`，然后再运行 PAV。
- PAV 从低分到高分维护块栈；只有左侧块概率严格大于右侧块概率时合并。等概率不合并，输出保持非递减。
- 输出是分段常数映射。相邻块的边界取“左块最后一个唯一观测分数”和“右块第一个唯一观测分数”的中点；输入分数等于阈值时归入左侧段。
- 区间外推由制品绑定：`clip` 使用最近端点概率并在 `uncertainties` 标记外推；`raise` 对区间外分数返回 `extrapolation_rejected`。
- 制品记录 `score_name`、`target_name`、训练分数范围、外推策略、训练指纹、块与合并路径。预测必须提交相同 `score_name`，评估还必须匹配 `target_name`。
- 训练集和评估集必须分开。评估集正权重记录的 SHA-256 指纹若与训练集完全相同，返回 `evaluation_dataset_overlap`。

## 模块关系

- `app/validation.py`：输入清洗、缺失/零权/负权/非有限值规则和失败类别。
- `app/kernel.py`：同分加权聚合与核心 PAV，输出块、合并事件和加权 SSE。
- `app/artifacts.py`：版本化分段常数模型制品、分段阈值、序列化与反序列化校验。
- `app/prediction.py`：分数到概率映射、分数含义绑定和外推策略。
- `app/evaluation.py`：评估集指纹分离检查、加权 Brier、log loss 和绝对误差。
- `app/repository.py`：SQLite 模型表和关联请求日志表。
- `app/services.py`：训练、预测、评估的业务编排。
- `app/api.py`：FastAPI 路由、请求身份、步骤版本、失败/警告/不确定结论响应。
- `tests/`：独立断言测试。参考 QP 使用 SciPy 直接优化加权二次目标，不用被测 PAV 生成期望。

## 本地运行

不需要创建虚拟环境，直接使用系统 `python3`：

```bash
python3 -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

健康检查：

```bash
curl -s http://127.0.0.1:8000/health
```

训练模型：

```bash
curl -s -X POST http://127.0.0.1:8000/v1/models \
  -H 'Content-Type: application/json' \
  -H 'X-Request-ID: demo-train-001' \
  -d '{
    "score_name": "risk_score",
    "target_name": "default_next_30d",
    "extrapolation": "clip",
    "records": [
      {"score": 0.1, "target": 0, "weight": 2.0},
      {"score": 0.4, "target": 1},
      {"score": 0.6, "target": 0},
      {"score": 0.9, "target": 1, "weight": 3.0}
    ]
  }'
```

将响应中的 `model_id` 用于：

- `POST /v1/predictions`
- `POST /v1/evaluations`
- `GET /v1/models`
- `GET /v1/request-logs/{request_id}`

SQLite 默认路径为 `data/calibration.db`，可用 `CALIBRATION_DB=/path/to.db` 覆盖。

## 验证命令

```bash
python3 -m pytest
python3 -m compileall -q app tests
```

预期判断：

- pytest 显示 `22 passed`。测试包含手算违背单调序列、三次连续合并路径、同分加权聚合、边界权重、零权过滤、独立加权损失公式、SciPy 约束 QP 对照、具体失败类别、制品序列化、外推、SQLite 日志和 HTTP 全链路。
- compileall 无输出且退出码为 0。
- 当前环境会出现一条 Starlette/httpx TestClient 的弃用 warning，不影响测试结果。

## 依赖版本

项目直接依赖：

- Python >= 3.10
- FastAPI
- NumPy
- Pydantic
- uvicorn
- SQLite（Python 标准库 `sqlite3`）

测试额外使用：

- pytest 9.1.1
- httpx 0.28.1
- SciPy 1.18.1

本机实际验证版本：

- FastAPI 0.141.1
- NumPy 2.5.3
- Pydantic 2.13.5
- uvicorn 0.54.0
- pytest 9.1.1
- httpx 0.28.1
- SciPy 1.18.1

已按约定直接使用全局安装包，没有创建 `.venv`，也没有安装新包。

## 响应可读性

所有响应都带 `request_id`，可由 `X-Request-ID` 指定。`steps` 展示处理模块、版本和关键动作；`failures`、`warnings`、`uncertainties` 分开。训练响应包含 `merge_events`，其中有每次合并的左右块分数、权重、概率、合并概率和加权损失增量。请求成功和失败都会写入 SQLite，并用相同 `request_id` 关联。
