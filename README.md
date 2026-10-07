# 小语料 Skip-gram 负采样训练后端

基于 Python + FastAPI + NumPy + SQLite 的本地小语料训练后端，提供词向量、
余弦相似度、最近邻和调用方自带标签的夹具评估。所有数据均为本地合成夹具，
不连接外部服务，不声称任何真实语义评测结论。

## 模块职责

| 文件 | 职责 |
| --- | --- |
| `sgns/errors.py` | 五类错误契约：`INPUT_ERROR`、`NOT_FOUND`、`STATE_CONFLICT`、`RESOURCE_EXHAUSTED`、`COMPUTATION_FAILED`。 |
| `sgns/config.py` | 不可变训练配置及校验。 |
| `sgns/data.py` | 语料字符串校验、句子切分、确定性小写分词、显式 token 预算。 |
| `sgns/vocab.py` | 词频统计、`min_count` 截断、词典冻结、子采样概率。 |
| `sgns/sampling.py` | `count**0.75` 逆 CDF 负采样；负例等于正目标时拒绝重抽。 |
| `sgns/kernel.py` | 稳定 sigmoid、单事件负采样更新、确定性训练循环、中间状态 trace。 |
| `sgns/artifact.py` | 输入/输出双向量 NPZ 制品、sha256 sidecar、加载校验。 |
| `sgns/storage.py` | SQLite 语料与训练运行状态（`PENDING/COMPLETED/FAILED`）。 |
| `sgns/evaluation.py` | 向量查询、余弦相似度、最近邻、自带标签夹具评估。 |
| `sgns/service.py` | 跨模块编排与状态机。 |
| `sgns/schema.py`, `sgns/app.py` | Pydantic 模型与 FastAPI HTTP 边界。 |
| `tests/reference.py` | 独立标量参考实现和手算常量（不导入训练内核）。 |
| `tests/replay_log.py` | 按固定 run id 写出的 JSONL 重放日志。 |

## 明确的算法约定

1. 词典冻结：训练前按原始词频统计；`count >= min_count` 保留，顺序为
   频次降序、同频按字典序。训练期间不再增删词。
2. 子采样：`keep(w)=min(1, sqrt(freq(w)/t)+freq(w)/t)`，`t=0` 关闭。
   每个 token 使用随机流中的一个新均匀数，严格小于 `keep` 时保留。
3. 负采样分布：`P(w)=count(w)**0.75 / sum(count**0.75)`，冻结后构建
   累积分布，逆 CDF 抽取。负例与正目标相同即丢弃并立即重抽（计一次
   redraw，预算 100 次，超限为 `COMPUTATION_FAILED`）；同一目标的负例
   允许重复，重复负例按同一更新前行累加。
4. 随机流：单个 `numpy.random.default_rng(seed)` 依次用于输入向量初始化、
   子采样、动态窗口 `[1,window]`、负采样；顺序固定为 epoch -> 句子插入序
   -> 中心位置 -> 上下文位置。输入向量初始化于 `[-0.5/dim, 0.5/dim)`，
   输出向量初始化为零。
5. 单事件更新：所有输出行先快照；正系数 `lr*(1-sigmoid)`，负系数
   `-lr*sigmoid`；输入向量与每个输出行分别更新，互不覆盖。
6. 数值稳定：sigmoid 分支实现，正负 700 不溢出；分数、系数、更新结果
   非有限时立即抛出 `COMPUTATION_FAILED`。
7. 制品：NPZ 同时保存输入与输出向量；`.npz.sha256` 校验摘要，摘要不符
   或版本不符为 `COMPUTATION_FAILED`。查询只使用输入向量。

## 错误语义与 HTTP 状态

| 错误码 | HTTP | 触发场景（示例） |
| --- | --- | --- |
| `INPUT_ERROR` | 400 | 空语料、非字符串、非法配置、非法评估标签。 |
| `NOT_FOUND` | 404 | 语料/运行不存在；词不在冻结词典。 |
| `STATE_CONFLICT` | 409 | 重复 run id；重复创建 corpus id；查询未完成/失败运行。 |
| `RESOURCE_EXHAUSTED` | 413 | 超过调用方声明的 `max_tokens`。 |
| `COMPUTATION_FAILED` | 422 | NaN/inf、零范数余弦、负采样重抽耗尽、制品摘要不符。 |

错误响应固定为：

```json
{"error": "NOT_FOUND", "message": "...", "details": {"word": "x"}}
```

## 运行方式

环境按约定使用系统 `python3`，无需创建虚拟环境，依赖已全局安装。

运行测试（实际执行并输出摘要，同时刷新 `test_runs/*.jsonl`）：

```bash
python3 -m pytest -q
```

本地演示（合成语料、训练、同种子复现、相似度、最近邻）：

```bash
python3 demo.py
```

启动服务：

```bash
./run_server.sh          # 默认 127.0.0.1:8000
./run_server.sh 8010     # 指定端口
```

HTTP 示例：

```bash
curl -s -X POST localhost:8000/corpora \
  -H 'content-type: application/json' \
  -d '{"corpus_id":"c1","text":"a b a c b\nc a b a c"}'

curl -s -X POST localhost:8000/training-runs \
  -H 'content-type: application/json' \
  -d '{"run_id":"r1","corpus_id":"c1","config":{"dim":8,"epochs":10,"seed":7,"subsample_t":0.0}}'

curl -s localhost:8000/training-runs/r1/vectors/a
curl -s -X POST localhost:8000/training-runs/r1/similarity \
  -H 'content-type: application/json' -d '{"word_a":"a","word_b":"b"}'
curl -s -X POST localhost:8000/training-runs/r1/neighbours \
  -H 'content-type: application/json' -d '{"word":"a","top_k":2}'
curl -s -X POST localhost:8000/training-runs/r1/evaluate \
  -H 'content-type: application/json' \
  -d '{"pairs":[{"word_a":"a","word_b":"b","label":1}]}'
```

端点：

- `GET /health`
- `POST /corpora`
- `POST /training-runs`（同步训练，支持 `record_events` 记录前若干事件）
- `GET /training-runs/{run_id}`
- `GET /training-runs/{run_id}/vectors/{word}`
- `POST /training-runs/{run_id}/similarity`
- `POST /training-runs/{run_id}/neighbours`
- `POST /training-runs/{run_id}/evaluate`

## 复现证据

独立测试（22 项）包含：

- 手算单步梯度：固定二维向量下分数 `[0.06,-0.14,0.38]`、sigmoid、
  正/负系数、输入向量与三个输出行的精确数值，对照 `tests/reference.py`
  的独立标量实现，而非被测内核自证。
- 双向量不误覆盖：未使用的输入行和中心输出行保持原值；重复负例在更新前
  快照上累加两次。
- 负采样分布：精确权重加固定种子 60,000 次抽样，误差容差 1%。
- 正目标碰撞：均匀二元词典下负例永不等于正目标，同种子重放完全一致。
- 子采样：手算 `f=1e-5,t=1e-3 -> keep=0.11` 并用 40,000 次固定随机流验证。
- 固定语料重复训练：不同 SQLite/制品目录下同 corpus+seed 得到逐位一致
  的输入/输出向量和统计；重复 run id 为 `STATE_CONFLICT`。
- 未知词为 `NOT_FOUND`；零范数/NaN/制品篡改/资源超限/状态冲突各自映射
  到不同错误类别。

每次测试运行都会在 `test_runs/<run_id>.jsonl` 写结构化记录，包含 run id、
判断点、期望值、观测值、关键中间状态（种子、抽样数、系数、统计、状态码）
和判断理由；失败类别通过稳定错误码记录，可直接用日志里的种子和输入重放。

## 未实现/边界

- 同步训练，不做后台队列、分布式训练或 GPU。
- 评估只对调用方提供的合成标签做一致率统计，不内置真实语义基准，也不
  宣称语言理解提升。
- 分词面向 ASCII 词和单字 CJK 的合成夹具，不做生产级多语种分词。
