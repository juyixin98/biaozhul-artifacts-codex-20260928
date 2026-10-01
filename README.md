# Exact Finite-Set NSGA Backend

这是一个本地合成数据驱动的多目标非支配排序后端，技术栈为 Python、FastAPI、NumPy、SciPy 和 Pydantic。它支持固定优化方向、重复向量的不同对象身份、增量插入、增量删除、精确数值以及可选的确定性网格量化。

## 工程结构

- `app/parsing.py`：请求值解析、固定方向转换、精确数值策略和可选容差量化。
- `app/core.py`：严格支配图、重复向量分组、初始全量比较、局部增量重排和删除恢复。
- `app/audit.py`：独立分层诊断、重复身份检查、数值不确定项说明。
- `app/service.py`：线程内状态、操作编排、响应序列化。
- `app/main.py`：FastAPI 入口、请求 ID 中间件、可解释日志和 HTTP 接口。
- `tests/`：单元、集成和独立朴素参考实现对照测试。
- `data/sample_problem.json`：本地二维混合方向样例。
- `scripts/load_sample.py`：向运行中的服务加载样例。

## 关键规则

默认 `numeric_settings.mode = "exact"`。整数和十进制字符串解析为精确有理数；JSON 浮点按其实际二进制值精确表示，不使用隐式 epsilon。

所有目标先按固定方向转换为“越大越优”：

- `min`：转换值为 `-objective`
- `max`：转换值为 `objective`

严格支配要求“所有目标不差，且至少一个目标严格更好”。完全相同的向量互不支配，但是不同候选 ID 保留为独立对象，并共享同一个向量组的支配关系。

可选 `mode = "quantized"` 必须显式给出正 `tolerance`。系统先把声明值按十进制、半偶舍入到固定网格，再在网格值上执行精确严格支配。这不会在比较层制造非传递关系；被舍入影响的结论会在响应和日志证据中作为 `uncertainty` 单独列出。

## 启动

不需要创建虚拟环境或安装依赖，直接使用系统 Python：

```bash
cd /home/admin/Downloads/xinbiaozhu/opp307/a
./run.sh
```

等价命令：

```bash
python3 -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

健康检查：

```bash
curl -s http://127.0.0.1:8000/health
```

另开终端加载样例：

```bash
python3 scripts/load_sample.py
```

样例的方向是 `[cost:min, reliability:max]`。手算层号为：

1. `low-cost`、`balanced`、`low-cost-copy`
2. `expensive`、`weak`

其中 `low-cost` 与 `low-cost-copy` 是两个对象身份，但属于同一重复向量组。

## HTTP 接口

创建问题：

```bash
curl -s -X POST http://127.0.0.1:8000/v1/problems \
  -H 'content-type: application/json' \
  -H 'x-request-id: demo-create' \
  -d '{
    "problem_id": "demo",
    "objectives": [
      {"name": "cost", "direction": "min"},
      {"name": "gain", "direction": "max"}
    ],
    "candidates": [
      {"id": "a", "objectives": ["1", "1"]},
      {"id": "b", "objectives": ["3", "1"]}
    ]
  }'
```

读取状态：

```bash
curl -s http://127.0.0.1:8000/v1/problems/demo
```

增量插入：

```bash
curl -s -X POST http://127.0.0.1:8000/v1/problems/demo/candidates \
  -H 'content-type: application/json' \
  -d '{"candidates":[{"id":"c","objectives":["2","2"]}]}'
```

删除并恢复后续层：

```bash
curl -s -X DELETE http://127.0.0.1:8000/v1/problems/demo/candidates/c
```

候选关系和逐维解释：

```bash
curl -s http://127.0.0.1:8000/v1/problems/demo/candidates/a
curl -s http://127.0.0.1:8000/v1/problems/demo/explain/a/b
```

所有响应包含 `request_id`、`version`、`stage`、`data`、`evidence`、`errors`、`uncertainty`。失败原因在 `errors` 中单列；数值近似说明在 `uncertainty` 中单列。

## 测试

运行：

```bash
python3 -m pytest -q
```

本机真实输出：

```text
......................                                                   [100%]
22 passed, 1 warning in 0.52s
```

测试覆盖：

- 二维混合方向手算层号。
- 重复向量共享支配关系但保留对象身份。
- 全 `max` 和高维退化数据。
- 增量插入导致的局部层号变化。
- 删除最后一个重复组身份后，下游层恢复。
- 删除重复点的一个身份后，其他身份继续共享关系。
- 缺失 `objectives`、缺失某一维目标、维数错误、重复候选 ID 的具体失败类别。
- 默认精确模式不使用隐式 epsilon。
- 量化模式仍然保持严格偏序，并单列不确定项。
- FastAPI 请求 ID、证据步骤、审计检查、错误响应和逐维解释。
- 独立朴素全量两两比较参考实现，参考代码不调用被测排序函数。
