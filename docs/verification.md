# 验证记录

验证时间：2026-10-06，Asia/Shanghai。

## 自动化测试

命令：

```bash
pytest -q
```

结果：

```text
18 passed, 1 warning in 1.81s
```

唯一警告来自当前环境中 FastAPI/Starlette 的 `httpx` TestClient 版本弃用提示，不影响测试结果。

测试覆盖：

- 真实 uvicorn HTTP 服务，而非进程内假调用。
- 参考服务对重复关闭、越序关闭、越序提交、已关闭取消均返回 409 `state_conflict`。
- 故障服务对重复关闭、草稿越序关闭、已关闭取消均错误返回 200。
- 参考服务完整 MBT 运行通过；故障服务完整运行失败且类别为 `state_conflict`。
- SQLite 中保留 run id、seed、scheduler、前后状态、观察状态、选择原因和判定理由。
- 缩减候选先重验模型前置条件，再用全新资源实际重放；草稿越序关闭从 2 步 episode 保持为“创建 + 关闭”，没有无效空序列。
- 独立重放脚本真实复现故障并以退出码 1 结束。
- 输入错误、状态冲突、资源耗尽、计算失败分别断言为不同类别。

## 手工 HTTP 复现

启动：

```bash
python3 -m uvicorn mbt_server.services.reference_app:app --host 127.0.0.1 --port 8101 --log-level warning
python3 -m uvicorn mbt_server.services.buggy_app:app --host 127.0.0.1 --port 8102 --log-level warning
MBT_DATABASE_PATH=data/manual_evidence.sqlite3 \
  python3 -m uvicorn mbt_server.api:app --host 127.0.0.1 --port 8000 --log-level warning
```

参考服务结果：

```json
{
  "run_id": 1,
  "status": "passed",
  "failure_category": null,
  "failure_step": null,
  "scheduler": "targeted",
  "seed": 342,
  "steps_executed": 14
}
```

故障服务的独立运行记录：

```text
run 2: failed, state_conflict, step 4, reject_close_closed（重复关闭）
run 3: failed, state_conflict, step 2, reject_close_draft（草稿越序关闭）
run 4: failed, state_conflict, step 4, reject_cancel_closed（已关闭取消）
```

run 2 的最小脚本：

```bash
python3 data/replay_run_2.py
```

输出末尾为：

```text
{"body": {"id": "order-9", "state": "draft"}, "status": 201}
{"body": {"id": "order-9", "state": "submitted"}, "status": 200}
{"body": {"id": "order-9", "state": "closed"}, "status": 200}
{"body": {"id": "order-9", "state": "closed"}, "status": 200}
expected status 409, got 200
```

脚本退出码为 1，证明它复现的是同一状态契约违背，而不是只展示接口响应。

## 证据文件

- SQLite：`data/manual_evidence.sqlite3`
- 最小重放：`data/replay_run_2.py`
- 输入模型：`fixtures/order_model.json`
- 依赖版本：`requirements.txt`
