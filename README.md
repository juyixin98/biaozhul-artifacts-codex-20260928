# DistStats 本地分布式统计聚合服务

这是一个从空目录实现的 Go 后端工程，用本地 HTTP 分片提交模拟分布式参与者，把每个分区的稳定可合并统计量聚合到 PostgreSQL。所有测试数据均在本地生成，不依赖真实业务数据或线上账号。

## 技术栈与版本

- Go: `1.23.4 linux/amd64`
- PostgreSQL: 本机实测 `16.15 (Ubuntu 16.15-0ubuntu0.24.04.1)`
- HTTP: Go 标准库 `net/http`
- PostgreSQL 驱动: `github.com/jackc/pgx/v5 v5.7.5`
- 数据库迁移: 启动时执行嵌入式 `migrations/0001_init.sql`

## 目录结构

- `internal/compute`: 稳定数值内核、加权矩、协方差、不同合并树、摘要校验
- `internal/protocol`: HTTP JSON 协议和摘要 DTO
- `internal/store`: PostgreSQL 连接、schema、作业/分片/运行记录查询
- `internal/service`: 事务编排、去重、状态机、重放和错误契约
- `internal/httpapi`: 标准库 HTTP handler、状态码映射、运行响应头
- `migrations`: SQL schema
- `scripts/init-db.sh`: 本地数据库创建脚本

## 本地配置

环境变量如下：

```text
DATABASE_URL=postgres:///opp298?sslmode=disable
HTTP_ADDR=:8080
MAX_DIMENSIONS=64
MAX_OBSERVATIONS=100000
```

创建数据库：

```bash
DB_NAME=opp298 DB_USER="$USER" ./scripts/init-db.sh
```

启动服务：

```bash
go run ./cmd/server
```

服务启动时自动建表。健康检查：

```bash
curl -s http://127.0.0.1:8080/healthz
```

## 请求样例

创建二维作业：

```bash
curl -s -X POST http://127.0.0.1:8080/v1/jobs \
  -H 'Content-Type: application/json' \
  -d '{"dimension":2}'
```

提交分片。`weight` 省略时表示权重 1，显式 `0` 是合法零权重观测；大偏移数据应让同一作业的分片使用接近数据的共同 `origin`：

```bash
JOB_ID='替换成作业ID'
curl -s -X POST "http://127.0.0.1:8080/v1/jobs/$JOB_ID/shards" \
  -H 'Content-Type: application/json' \
  -d '{
    "shard_id":"north",
    "origin":[10,20],
    "observations":[
      {"values":[11,21]},
      {"values":[13,19],"weight":1},
      {"values":[99,99],"weight":0}
    ]
  }'
```

也可以提交已由外部 worker 计算好的 summary；协议字段见 [docs/protocol.md](docs/protocol.md)。重复 `(job_id, shard_id)` 返回 409，不会重复合并。

重放并比较顺序合并树与平衡合并树：

```bash
curl -s "http://127.0.0.1:8080/v1/jobs/$JOB_ID/replay?order=balanced&compare=true"
```

查询当前结果：

```bash
curl -s "http://127.0.0.1:8080/v1/jobs/$JOB_ID/result"
```

终结作业。终结后拒绝新分片：

```bash
curl -s -X POST "http://127.0.0.1:8080/v1/jobs/$JOB_ID/finalize"
```

所有写请求和重放请求都返回：

```text
X-Run-ID
X-Run-Number
```

使用运行编号读取可审计日志：

```bash
curl -s "http://127.0.0.1:8080/v1/runs/$RUN_NUMBER"
```

日志包含路由、请求、错误类别、阶段、判断理由、当时摘要状态；提交分片的关键事件和状态更新在同一个数据库事务内写入。

## 统计定义

权重模型是频率权重/质量权重：

- 权重必须是有限非负浮点数。
- 省略 `weight` 等于 1。
- `w=0` 合法，但不计入 `n`、`weight` 或矩。
- 总体协方差分母是 `W=sum(w)`。
- 样本协方差分母是 `W-1`，因此 `W<=1` 时样本协方差为 `null`。
- 空分片是单位元：`n=0,w=0`，总体和样本协方差均未定义。
- 单正权重样本的总体方差为 0，样本方差未定义。

内核不使用 `E[X^2]-E[X]^2`。它使用加权 Welford/Chan 风格的均值、一阶矩和 `M2` 矩阵，并在批处理路径使用本地原点与 Neumaier 补偿。

## 错误契约

HTTP body 中始终有 `error.kind`、`error.code`、`error.message`：

- `INVALID_INPUT`: 400，维度不兼容、非法权重、非法摘要等
- `STATE_CONFLICT`: 409，重复分片或终结后继续提交
- `NOT_FOUND`: 404，作业或运行不存在
- `RESOURCE_EXHAUSTED`: 422，维度/观测数/数据库资源限制
- `COMPUTATION_FAILED`: 422，权重或矩溢出
- `UNAVAILABLE`: 503，数据库或内部依赖不可用

## 测试命令

```bash
go test ./... -v
go test -race ./...
go vet ./...
```

高精度测试在 `internal/compute/summary_test.go` 中使用 `math/big.Float`（256 bit）独立计算参考答案；参考值不是由被测内核生成。测试会记录大偏移小方差、极端不均衡分片、极端权重、顺序/平衡合并树的实测误差。

本次实际执行结果见 [VERIFICATION.md](VERIFICATION.md)。
