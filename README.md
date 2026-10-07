# Isotonic Calibration Service

预测分数 -> 概率的保序回归（isotonic regression）校准后端，核心为加权
PAV（Pool-Adjacent-Violators）算法。技术栈：Python 3.12、FastAPI、
NumPy、SQLite（制品存储）、pytest + scipy（独立参考验证）。

## 模块关系

```
calibration_service/app/
  config.py        配置（DB 路径、默认 score 字段、ECE 分箱数；环境变量覆盖）
  validation.py    数据校验与同分合并（显式零权/缺失规则）
  pav.py           加权 PAV 训练内核（纯算法，无 IO）
  calibrator.py    拟合 + 分段常数预测 + 显式外推策略
  artifact.py      SQLite 模型制品存储（版本化，绑定 score 含义）
  evaluation.py    独立评估集指标（log_loss / brier / ECE）
  logging_setup.py 请求级日志（每行带 request_id）
  main.py          FastAPI 入口：/fit /predict /evaluate /models /health
tests/             独立测试（手算用例 + scipy QP 参考实现）
```

数据流：`/fit` -> validation（校验/零权剔除/同分加权合并）-> pav（块合并）
-> calibrator（阈值+概率阶梯）-> artifact（SQLite 持久化，版本递增）。
`/predict` 与 `/evaluate` 从制品加载映射；评估集与训练集由调用方分开提交。

## 算法假设与规则

- 目标：最小化加权平方误差 `sum w_i (y_i - g_i)^2`，约束 `g` 非递减；
  标签为 0/1，因此拟合值即校准概率（裁剪到 [0,1]）。
- 相同分数先按权重合并：权重相加，标签取加权平均。
- 权重为 0 的行被丢弃并在响应/日志中计数；权重为负或非有限 -> 422。
- score 非有限（NaN/inf）或标签非 0/1 -> 422，响应 `detail` 列出行号。
- 输出为分段常数映射：块 k 覆盖区间 `(edge_{k-1}, edge_k]`。
- 区间外推策略固定为 `clamp_to_boundary`：低于训练范围取首块概率，
  高于范围取末块概率，随制品一起持久化。
- 制品绑定 `score_field`（输入分数含义，默认 `model_score`，可用
  环境变量 `CALIBRATION_SCORE_FIELD` 或请求字段覆盖）。

## 运行

```bash
cd /home/admin/Downloads/xinbiaozhu/opp339/a
uvicorn calibration_service.app.main:app --port 8339
```

```bash
# 训练（手算可核验：0.3/0.4 处标签违反单调，合并为 0.5）
curl -s -X POST localhost:8339/fit -H 'content-type: application/json' -d '{
  "model_id": "demo",
  "dataset": {"scores": [0.1,0.2,0.3,0.4,0.5], "labels": [0,0,1,0,1]}
}'
# 预测（越界分数被钳制到边界概率）
curl -s -X POST localhost:8339/predict -H 'content-type: application/json' -d '{
  "model_id": "demo", "scores": [-1.0, 0.35, 0.45, 7.0]
}'
# 评估（独立评估集）
curl -s -X POST localhost:8339/evaluate -H 'content-type: application/json' -d '{
  "model_id": "demo",
  "dataset": {"scores": [0.15,0.35,0.45], "labels": [0,1,0]}
}'
```

预期：`/fit` 返回 `probabilities == [0.0, 0.0, 0.5, 1.0]`、
`monotone_non_decreasing == true`；`/predict` 返回
`[0.0, 0.5, 1.0, 1.0]`；`/evaluate` 返回 `brier_score ≈ 0.41667`。
每个响应带 `request_id`，服务日志每行含同一 `request_id`，
失败响应单列 `category`（validation_error / not_found）与失败行号。

## 验证

```bash
python3 -m pytest tests/ -q
```

预期 `32 passed`。测试包含：手算反单调序列与级联合并路径、重复分数
加权合并、零权/负权/NaN 边界、与 scipy SLSQP 约束二次优化参考实现
对比拟合值与加权损失（参考实现与被测核心不共享代码）、API 端到端的
具体数值断言与失败类别断言。

## 依赖版本（本机已全局安装）

fastapi 0.141.1 / numpy 2.5.3 / scipy 1.18.1 / pydantic 2.13.5 /
pytest、uvicorn、httpx（TestClient 用）。无需 PostgreSQL 或外部账号；
制品存储使用本地 SQLite 文件（默认 `./calibration_artifacts.db`，
环境变量 `CALIBRATION_DB` 覆盖）。
