# 本地批处理推测副本执行控制

这是一个从空目录实现的 Go 工程，用来模拟本地批处理分片在多个尝试并行执行时，如何保证“同一逻辑分片只有一个获胜产物发布”。HTTP 服务只使用 Go 标准库，状态与协调信息保存在 PostgreSQL，合成工作者和输入夹具都在本仓库内。

## 核心不变量

- `shards` 表示逻辑分片，`attempts` 表示该分片的原始尝试或推测尝试。
- 发布产物时在一个 PostgreSQL 事务中锁定逻辑分片、校验尝试仍处于活跃状态、插入唯一已发布产物、更新分片为 `succeeded`，随后将同一分片的其他活跃尝试标记为 `superseded`。
- 迟到失败或迟到产物不会覆盖赢家：终端分片上的回调会被记录为 `superseded`，临时文件会被删除。
- 推测触发受 `speculate_after_ms`、`max_attempts_per_shard` 和 `max_active_attempts` 约束，并写入 `SPECULATION_BUDGET` 或 `SHARD_ATTEMPT_STARTED` 事件。
- 作业只有在全部逻辑分片为 `succeeded` 时才进入 `succeeded`；任一分片在预算耗尽后失败，作业进入 `failed`。
- 协调器每次启动生成 epoch。重启恢复会把旧 epoch 的未完成尝试标记为 `interrupted`，旧回调会被 fence，避免重启后双发。
- 合并结果按 `artifacts(shard_id)` 唯一约束和有序查询输出，每个逻辑分片最多一条记录。

## 目录结构

- `cmd/coordinator`：HTTP 协调器、迁移、调度循环和优雅退出。
- `cmd/worker`：标准库 HTTP 合成工作者。
- `internal/protocol`：任务、分片、尝试、状态和失败分类协议。
- `internal/store`：PostgreSQL schema 迁移、事务状态机、预算调度、租约恢复和结果合并。
- `internal/coordinator`：标准库 HTTP API、临时产物登记、回放接口。
- `internal/scheduler`：轮询待调度尝试并调用本地工作者。
- `internal/worker`：计算内核、心跳、延迟/失败合成夹具和工作者 HTTP 服务。
- `internal/participant`：工作者回调 HTTP 客户端协议。
- `internal/artifacts`：临时产物 checksum 校验、不可变发布文件和回收。
- `migrations`：PostgreSQL schema。
- `testdata/fixtures`：独立的最小数据夹具和预期输出。
- `integration`：真实 PostgreSQL 集成测试。
- `examples`：提交请求和服务调用示例。

## 失败分类

- `VALIDATION`：请求、checksum 或身份信息不合法。
- `COMPUTE`：合成计算失败或未知计算错误归一化后的类别。
- `WORKER_UNAVAILABLE`：协调器无法把尝试派发给工作者。
- `LEASE_EXPIRED`：工作者心跳未续租。
- `INTERRUPTED`：协调器重启或 epoch fence。
- `SUPERSEDED`：另一个尝试已经发布该逻辑分片。

未知异常不会统一返回成功。HTTP 层会保留具体错误类别和消息，状态机也会把类别写入尝试与事件。

## 本地准备

要求：

- Go 1.24+
- PostgreSQL 14+
- `psql`
- `jq`、`curl`（仅演示脚本需要）

默认连接本机 Unix socket，避免 TCP 密码认证差异。创建开发和测试库：

```bash
make setup-db
```

如果 PostgreSQL 超级用户连接方式不同，可以覆盖：

```bash
ADMIN_PSQL='psql postgresql://postgres@127.0.0.1:5432/postgres?sslmode=disable' scripts/setup-db.sh
```

复制配置：

```bash
cp config/dev.env.example .env
set -a
. ./.env
set +a
```

## 启动服务

终端 1：

```bash
go run ./cmd/coordinator
```

终端 2：

```bash
go run ./cmd/worker
```

终端 3 提交演示：

```bash
examples/http-calls.sh examples/speculative-failure-job.json
```

也可以提交普通慢速分片：

```bash
examples/http-calls.sh examples/submit-job.json
```

脚本会打印作业 ID、版本化进度、尝试状态、事件流和合并结果。协调器与工作者日志是 JSON 或测试中的文本日志，包含 `job_id`、`shard_key`、`attempt_id`、`epoch`、版本、延迟、心跳和判定依据。

## HTTP 接口

- `POST /v1/jobs`：提交作业。
- `GET /v1/jobs/{id}`：作业状态与版本。
- `GET /v1/jobs/{id}/progress`：可解释进度和活跃尝试预算。
- `GET /v1/jobs/{id}/shards`：逻辑分片状态。
- `GET /v1/jobs/{id}/attempts`：所有尝试及失败类别。
- `GET /v1/jobs/{id}/events?after_id=0&limit=1000`：事件回放。
- `GET /v1/jobs/{id}/result`：按分片序号合并的结果。
- `POST /v1/jobs/{id}/replay`：基于原作业不可变输入创建独立重跑。
- `POST /v1/attempts/{id}/start`
- `POST /v1/attempts/{id}/heartbeats`
- `POST /v1/attempts/{id}/artifacts?checksum_sha256=...&epoch=...`
- `POST /v1/attempts/{id}/complete`
- `POST /v1/attempts/{id}/fail`

## 测试

运行快速单元测试：

```bash
make test
```

运行真实 PostgreSQL 集成测试：

```bash
make integration
```

一键运行全部检查并保存可复核输出：

```bash
scripts/run-tests.sh
```

启动真实二进制、自动准备临时 PostgreSQL 库并运行正常/异常两个夹具：

```bash
scripts/smoke.sh
```

输出保存在：

- `test-results/unit-tests.out`
- `test-results/integration-tests.out`
- `test-results/go-vet.out`

集成测试覆盖：

- `TestSpeculativeFailureThenSlowOriginalPublishesExactlyOnce`：原始尝试慢，推测副本先失败；断言最终赢家是原始尝试，失败类别是 `COMPUTE`，产物只发布一次，临时目录为空。
- `TestConcurrentCompleteRequestsPublishOnlyOneWinner`：同一逻辑分片两个尝试同时提交产物；断言一个 200、一个 409 `SUPERSEDED`，合并结果只有一个分片。
- `TestCoordinatorRestartFencesOldAttemptAndReplaysJobToCompletion`：等待旧尝试运行后重启协调器；断言旧尝试为 `interrupted`，新尝试为新 epoch 的 replacement original，作业不重复完成。
- `TestReplayUsesStoredInputsAndIndependentRun`：从事件/输入创建独立重跑，不重写原作业。

夹具中的预期结果位于 `testdata/fixtures/*.json`，不由被测核心生成。测试读取这些固定预期并断言输出、失败类别、版本、事件、产物数量和资源回收。

## 依赖锁定

运行：

```bash
go mod download
```

依赖版本由 `go.mod` 和 `go.sum` 锁定。生产代码的外部依赖只有 `github.com/jackc/pgx/v5`；HTTP 路由、JSON、工作者和调用脚本均使用标准库或系统工具。
