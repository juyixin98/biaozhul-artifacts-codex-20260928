# 本地双输入流算子：对齐 / 非对齐检查点教学框架

这是一个从空工程搭建的 Go 后端，用可运行的双输入流算子演示并验证：

- 对齐检查点：先到屏障会阻塞对应输入边界，但队列有明确背压上限。
- 非对齐检查点：快照保存基础状态和明确的在途记录集合，恢复时只重放未纳入快照状态的记录。
- 单调屏障 ID：同一任务内屏障 ID 必须递增；当前支持一个 durable snapshot 在途。
- 输出提交顺序：checkpoint 与输出明细在 PostgreSQL 中同一事务提交，输出带确定性 `output_id` 和屏障 epoch，恢复后可幂等重放。
- 可解释接口：每个请求有 `X-Request-ID`，响应列出关键步骤，失败单独给出类别；存储失败标记为不确定结论。

## 模块职责

- `internal/protocol`：任务配置、记录、屏障、checkpoint、状态和指标协议。
- `internal/kernel`：有界队列、屏障状态机、双路 keyed 计数内核、在途缓冲与快照生命周期。
- `internal/storage`：内存存储和 PostgreSQL 存储；checkpoint、状态 JSON、在途 JSON、输出明细在一个事务中落盘。
- `internal/source`：合成两路输入、速率配置、顺序校验和恢复位点跳过。
- `internal/replay`：从 checkpoint 重建状态，只补放 `inflight` 中显式命名的记录。
- `internal/engine`：运行合成任务、模拟 CP2 快照中断，再恢复到新的 CP2。
- `internal/httpapi`：标准库 `net/http` 接口、请求身份、错误类别和结构化日志。
- `migrations/001_checkpoints.sql`：PostgreSQL 表结构。
- `samples/two-speed.json`：可直接使用的两路速率差样例。

## 首次运行

要求 Go 1.23+。内存模式不依赖外部服务：

```bash
make run
```

健康检查：

```bash
curl -i http://127.0.0.1:8099/healthz
```

运行一次非对齐实验：

```bash
curl -sS -X PUT http://127.0.0.1:8099/v1/tasks/demo-two-speed/workload \
  -H 'Content-Type: application/json' \
  --data-binary @samples/two-speed.json

curl -sS -X POST http://127.0.0.1:8099/v1/tasks/demo-two-speed/runs \
  -H 'Content-Type: application/json' \
  -H 'X-Request-ID: lab-run-1' \
  -d '{"mode":"unaligned","backpressure_limit":4,"max_concurrent_checkpoints":1}' | jq

curl -sS http://127.0.0.1:8099/v1/tasks/demo-two-speed/checkpoints/latest | jq
```

恢复接口：

```bash
curl -sS -X POST http://127.0.0.1:8099/v1/tasks/demo-two-speed/recover \
  -H 'Content-Type: application/json' \
  -d '{"backpressure_limit":4,"max_concurrent_checkpoints":1}' | jq
```

## PostgreSQL

默认 PostgreSQL 配置使用本机 Unix socket：

```text
host=/var/run/postgresql dbname=opp299 user=admin sslmode=disable
```

首次准备当前机器上的数据库：

```bash
sudo -u postgres createdb -O admin opp299
sudo -u postgres psql -d opp299 -c 'GRANT ALL ON SCHEMA public TO admin;'
make postgres-run
```

服务启动时自动执行 `migrations/001_checkpoints.sql`。也可以用 `DATABASE_URL`、`HTTP_ADDR` 和 `STORAGE` 覆盖配置。

## 检查点语义

### 对齐模式

1. 屏障是带外控制事件，但同一输入中不能越过屏障之前的数据。
2. 第一个屏障到达后，算子只提交该屏障之前的同流前缀并等待另一路屏障。
3. 第二个屏障到达时，所有屏障之前的数据都已经纳入状态；checkpoint 的 `inflight` 必须为空。
4. 屏障之后的记录不进入该 checkpoint，快照 durable 前不会作为该 epoch 的结果提交。

### 非对齐模式

1. 第一个屏障到达时，先应用本输入屏障前的 FIFO 前缀。
2. 另一路输入中已经排队、但未纳入状态的记录进入 `inflight`，记录顺序和流身份被保存。
3. 第二个屏障到达时，快照状态不包含这些在途记录；恢复先用独立 replay 步骤补放它们。
4. 源恢复位点是每路 `BarrierAfter`；因此已经在源位点之后的记录不会再次从源日志读取。

`max_concurrent_checkpoints` 在教学实现中明确只支持 `1`。配置为更大值会返回 `protocol_violation`，避免把“多个并发快照的状态所有权”隐藏在代码里。

## 测试

普通单元与 HTTP 集成测试：

```bash
make test
```

包含 PostgreSQL 的集成测试：

```bash
make integration
```

竞态检测：

```bash
make race
```

测试覆盖的关键断言：

- `TestBoundedQueueEnforcesConfiguredBackpressure`：队列容量为 2 时第三个 producer 阻塞，消费后恢复。
- `TestAlignedCheckpointDrainsQueuesAndBlocksBehindBarrier`：对齐 CP 后在途为 0，屏障后记录不会提前进入状态。
- `TestUnalignedSnapshotCarriesOnlyUnincorporatedRecordsAndRecoveryReplaysThem`：基础状态、在途集合和恢复计数使用测试内字面 oracle，不由被测内核生成。
- `TestSnapshotInterruptionLeavesNoCheckpointAndAllowsRecoveryFromPreviousCompleteOne`：CP2 存储中断后只能从完整 CP1 恢复。
- `TestAlignedAndUnalignedRecoveryProduceSameReferenceCountsWithDifferentBuffers`：两种模式最终引用计数相同，但在途数量不同，并断言不重复计算。
- `TestHTTPRunExplainableResultAndFailure`：请求身份、步骤、状态结果和失败类别都出现在响应中。
- `TestPostgresAtomicCheckpointAndOutputs`：验证 PostgreSQL 原子写入、输出明细和旧 checkpoint 拒绝。

## 本机验证结果

以下命令在当前工程执行通过：

```text
$ go test ./... -count=1 -timeout 10s
ok  	opp299/internal/engine
ok  	opp299/internal/httpapi
ok  	opp299/internal/kernel
ok  	opp299/internal/replay
ok  	opp299/internal/source

$ go test -race ./... -count=1 -timeout 20s
ok  	opp299/internal/engine
ok  	opp299/internal/httpapi
ok  	opp299/internal/kernel
ok  	opp299/internal/replay
ok  	opp299/internal/source

$ go test -tags integration ./... -count=1 -timeout 20s
ok  	opp299/internal/httpapi
ok  	opp299/internal/kernel
ok  	opp299/internal/replay
ok  	opp299/internal/source
ok  	opp299/internal/storage
```

标准 HTTP 冒烟使用 `samples/two-speed.json` 跑两种模式，最终引用计数一致：

```text
aligned   k1={A:3 B:15}, k2={A:20 B:100}, buffer_occupancy=0
unaligned k1={A:3 B:15}, k2={A:20 B:100}, latest cp inflight.B=2
recover   replay=2, final state k1={A:3 B:15}, k2={A:20 B:100}
```

## HTTP 失败类别

- `protocol_violation`：屏障乱序、重复、跳号、配置不合法。
- `capacity_exceeded`：超过有界队列容量时的调用方语义。
- `storage_failure`：checkpoint 事务失败；结果标记为不确定，恢复必须选择最新完整 checkpoint。
- `cancelled`：请求或运行上下文被取消。
- `not_found`：任务 workload 或 checkpoint 不存在。
