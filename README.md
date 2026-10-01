# MCMC Diagnostics Service

本地 FastAPI 后端，用于标量单链及多链数值样本的：

- 链内自相关（ACF）
- 有效样本量（ESS）
- 批内/批间方差分离
- 经典 R-hat 与 rank-normalized split R-hat
- 常数链、短链、单链、错位均值链等边界诊断

运行时依赖为 Python、FastAPI、NumPy、SciPy、SQLite（Python 标准库）。独立测试额外使用 Statsmodels 与 ArviZ 作为成熟统计实现参照。

## 模块职责

- `mcmc_diagnostics/models.py`：Pydantic 样本契约和错误契约。
- `mcmc_diagnostics/kernel.py`：纯统计内核；ACF、方差分量、ESS、R-hat 都在这里实现。
- `mcmc_diagnostics/versioning.py`：版本信封、请求关联 ID、SQLite 表和结果/事件持久化。
- `mcmc_diagnostics/main.py`：FastAPI 路由、请求日志、错误分类和依赖注入。
- `mcmc_diagnostics/config.py`：本地环境变量配置。
- `mcmc_diagnostics/experiments.py`：固定 seed 的合成夹具、复现实验和独立参考库比较。
- `tests/`：独立统计测试、边界类别测试、HTTP 与 SQLite 端到端测试。
- `experiments/report.json`：实际执行复现实验生成的报告。

## 统计规则

### ACF

每条链分别去均值：

```text
x_demeaned[m, t] = x[m, t] - mean(x[m, :])
```

ACF 使用常见的 `1/N` 有偏自协方差估计，并除以该链 lag 0 自协方差：

```text
C(k) = sum_{t=0}^{N-1-k} x_demeaned[t] * x_demeaned[t+k] / N
ACF(k) = C(k) / C(0)
```

因此常数链没有可归一化的非零分母，服务返回 `acf = null` 和 `CONSTANT_CHAIN`，不会把重复值伪装成可解释 ACF。

接口中的 `max_acf_lag` 只控制展示的 ACF 长度，不控制 ESS 的相关项截断。

### ESS

ESS 不把 ACF 截到固定 lag 后直接求和。内核先分别进行链内去均值，再在多链层面计算 Stan/BDA3 风格的池化相关系数：

```text
ESS = M*N / tau
tau = -1 + 2 * sum(P_t)
P_t = rho(2t) + rho(2t+1)
```

截断使用两层规则：

1. Geyer initial positive sequence：成对相关和非正后停止。
2. Geyer initial monotone sequence：后续成对和不得高于前一对，非增部分被收缩。

服务返回三种估计：

- `identity_original_chains`：原始链上的 ESS，适合直接检验时间退化。
- `mean_split_chains`：每链从中间拆成两段后估计，可检测链前半/后半漂移。
- `bulk_split_rank_normalized`：拆分后联合 rank-normal transform，再估计 bulk ESS。

全常数输入的 ESS 不可估计，返回 `not_estimated_constant`。注意：ArviZ 对全常数输入有返回样本总数的库约定；复现实验会记录这个约定，但本服务不把它解释成“有效样本等于名义样本”。

### 方差与 R-hat

每条链均值单独返回，并区分：

- `within_batch_variance_W`：各链无偏方差（`ddof=1`）的均值。
- `between_batch_variance_B_scaled_by_n`：链均值方差（`ddof=1`）乘以每链长度 `N`。
- `marginal_variance_estimate_V_hat`：BDA3 混合方差估计。

服务返回：

- `classic_original_chains`：原始多链经典 potential scale reduction。
- `rank_split_chains`：每链拆分后，对 rank-normal 值和 folded rank-normal 值分别计算，再取最大值。
- `within_chain_rank_split_for_single_chain`：单链时仅作为链内前后半漂移检查；它不能替代多条独立链之间的分歧诊断。

常数但均值错位的链会有 `within=0, between>0`，数学上 R-hat 趋向正无穷。服务不写入非标准 JSON `Infinity`，而是返回：

```json
{"value": null, "extended_value": "plus_infinity", "status": "inconclusive_zero_within_variance"}
```

## 本地运行

不需要创建虚拟环境；本机约定直接使用 `python3`。

```bash
cd /home/admin/Downloads/xinbiaozhu/opp315/a
MCMC_DB_PATH=./data/runtime.db uvicorn mcmc_diagnostics.main:app --host 127.0.0.1 --port 8011
```

健康检查：

```bash
curl http://127.0.0.1:8011/healthz
```

提交诊断：

```bash
curl -sS http://127.0.0.1:8011/api/v1/diagnostics \
  -H 'Content-Type: application/json' \
  -H 'X-Request-ID: demo-iid' \
  --data @examples/request_iid.json
```

读取持久化结果和事件：

```bash
curl http://127.0.0.1:8011/api/v1/results/demo-iid
curl http://127.0.0.1:8011/api/v1/results/demo-iid/events
```

## 复现实验

```bash
python3 -m mcmc_diagnostics.experiments --output experiments/report.json
```

实验固定生成：

- 独立噪声链：ESS 应接近名义样本数。
- AR(1), `phi=0.8`：理论平稳自相关在 lag 1 约为 0.8，ESS/N 约为 `(1-phi)/(1+phi)=1/9`。
- 错位均值链：ESS 很低，rank split R-hat 显著大于 1。
- 全常数链：专门返回不可估计/不确定类别。
- 短链：禁止拆分 ESS 和 split R-hat，给出短链类别。

夹具由实验文件中的 NumPy 公式独立生成；测试参考答案来自 Statsmodels/ArviZ，不由被测 `kernel.py` 生成。

## 测试

```bash
pytest -q
```

测试会断言具体数值、具体失败类别、成熟库一致性、API 状态码、请求关联和 SQLite 持久化，而不仅验证接口能调用。

## 配置

可通过环境变量覆盖：

- `MCMC_DB_PATH`：SQLite 文件路径，默认 `./data/diagnostics.db`。
- `MCMC_MAX_CHAINS`：最大链数，默认 100。
- `MCMC_MAX_DRAWS`：每链最大样本数，默认 100000。
- `MCMC_MAX_ACF_LAG`：默认 ACF 最大展示 lag，默认 50。
- `MCMC_RHAT_WARNING_THRESHOLD`：R-hat 告警阈值，默认 1.01。
- `MCMC_ESS_LOW_FRACTION`：ESS/名义样本数的低比例阈值，默认 0.10。

## 局限

- R-hat 接近 1、ESS 不小，不等于已经证明收敛；只能说明这些诊断没有从给定窗口中发现强烈混合失败证据。
- 如果所有链都错过同一个模态，或样本窗口之后才出现非平稳，诊断可能无法发现。
- 本服务只支持一维标量链；多变量参数需逐参数调用。
- ESS 是积分自相关时间的有限样本估计，本身有抽样噪声；有限样本下也可能因负相关得到略大于名义样本数的 ESS，服务不强行截断。
- 单链无法真正评估多条独立轨迹之间的分歧，必须结合分散初始化的多条链。
