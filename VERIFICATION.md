# 验收记录

执行日期：2026-09-29（Asia/Shanghai）

## 环境

- Go: `go1.23.4 linux/amd64`
- PostgreSQL: `psql 16.15 (Ubuntu 16.15-0ubuntu0.24.04.1)`，本地 5432
- 数据库: `postgres:///opp298?sslmode=disable`
- pgx: `github.com/jackc/pgx/v5 v5.7.5`

数据库由以下命令创建：

```bash
DB_NAME=opp298 DB_USER=admin ./scripts/init-db.sh
```

实际输出：

```text
DO
CREATE DATABASE
PostgreSQL database opp298 is ready for user admin
```

## 自动化测试

执行：

```bash
go test ./... -v
go vet ./...
go test -race ./...
```

结果：全部通过。

关键数值测试的实际测量：

| 场景 | 顺序树误差 | 平衡树误差 | 容差 |
| --- | --- | --- | --- |
| 大偏移小方差 `1e15 + 小残差` | mean `0`, M2 `2.2737367544323206e-13` | mean `0`, M2 `2.2737367544323206e-13` | mean `100000.000000001`, M2 `4.104e-7` |
| 极端不均衡分片 `1/2/1998` | mean `0.0001220703125`, M2 `0` | mean `0.0001220703125`, M2 `0` | mean `66.33186027440605`, M2 `5.1758884453449024e+17` |
| 极端权重 `1e-12/1e-6/1e6/0.25` | mean `0`, M2 `9.094947017729282e-13` | mean `0`, M2 `9.094947017729282e-13` | mean `1.1099996225000845e-8`, M2 `5.800248575978182e-7` |

不同合并树直接比较：

```text
max mean error = 0
max M2 error   = 1.1102230246251565e-16
```

参考值由测试中的 `math/big.Float`（256 bit）两遍统计独立计算，不经过被测内核。

HTTP/PostgreSQL 集成测试覆盖：

- 创建作业、提交两个分片、平衡树重放
- 重复 shard ID 返回 `STATE_CONFLICT`，且失败事件进入运行日志
- 维度不兼容返回 `INVALID_INPUT/DIMENSION_MISMATCH`
- 维度超限返回 `RESOURCE_EXHAUSTED`
- 非法 JSON 返回 `INVALID_INPUT/MALFORMED_JSON`
- 不存在作业返回 `NOT_FOUND`
- 权重溢出返回 `COMPUTATION_FAILED`
- 作业 finalize 后再提交返回 `STATE_CONFLICT`
- 可通过运行编号读取阶段、判断理由和中间状态

## 手工 HTTP 验证

启动：

```bash
go build -o /tmp/diststats-server ./cmd/server
HTTP_ADDR=:18080 DATABASE_URL='postgres:///opp298?sslmode=disable' /tmp/diststats-server
```

健康检查：

```json
{"status":"ok"}
```

手工提交两个二维分片后，`GET /v1/jobs/{id}/replay?order=balanced&compare=true` 的实际关键结果：

```json
{
  "shard_count": 2,
  "means": [11.5, 20.5],
  "population_covariance": [1.25, -0.25, -0.25, 1.25],
  "sample_covariance": [
    1.6666666666666667,
    -0.3333333333333333,
    -0.3333333333333333,
    1.6666666666666667
  ],
  "comparison": {
    "reference_order": "sequential",
    "max_mean_error": 0,
    "max_m2_error": 0,
    "max_weight_error": 0,
    "tolerance": 1.205e-8,
    "accepted": true
  }
}
```

首次分片响应头：

```text
X-Run-Id: c319b8b1-310f-4ef5-afb3-66b7e79028a9
X-Run-Number: 178
```

验证后测试服务已停止。
