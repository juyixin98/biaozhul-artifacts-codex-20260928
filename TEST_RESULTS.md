# 本地验证记录

执行时间：2026-09-30，Asia/Shanghai 时区。

## 自动化测试

命令：

```bash
pytest -q
```

结果：

```text
13 passed, 1 warning in 0.38s
```

失败：0。

跳过/未执行：0。

唯一警告来自当前环境中的 `starlette.testclient`：

```text
StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated;
install `httpx2` instead.
```

该警告不影响测试结果；项目仍按 `requirements.lock` 使用当前可运行的 FastAPI、Starlette 与 HTTPX 精确版本。

## 独立参考对照

命令：

```bash
python3 scripts/verify_with_reference.py
```

结果：退出码 0。脚本输出合法 JSON，覆盖：

- 边界最优：SciPy SLSQP 参考点误差约 `7.77e-15`。
- 冗余约束：SciPy SLSQP 参考点误差约 `7.77e-15`。
- 近退化：SciPy SLSQP 参考点误差约 `3.33e-16`。
- 秩亏等式：解析参考点 `[0.5, 0.5]`，误差约 `1.11e-16`。
- 无可行解：HiGHS 状态码 2，服务返回 `infeasible`。

## HTTP 冒烟

本机 8000、8001、8010 端口启动时已被占用，因此使用 18765：

```bash
python3 -m uvicorn app.main:app --host 127.0.0.1 --port 18765
```

已验证：

- `GET /health` 返回 200、版本和依赖信息。
- `POST /api/v1/qp/solve` 使用 `examples/boundary_qp.json` 返回 `optimal`、`x=[0,1]`、活动约束和机器精度 KKT 残差。
- 错误形状的 `objective.linear` 返回 400、`linear_dimension_mismatch`、明确原因和请求 ID。
