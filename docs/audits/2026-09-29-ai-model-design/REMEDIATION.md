# ai-model-design v2 协议修复与复验

修复日期：2026-09-30。协议基线仍为 ai-model-design 文档提交 `358ffc407439657131be5b71212fc8afa6973a73`。本文件补充原始 REPORT.md，不覆盖修复前的失败证据。

## 修复范围

| 原问题 | 修复 |
|---|---|
| F01 必需预处理器被忽略 | 遍历所有部件；不支持的必需 pipeline/preprocessor 明确返回 MODEL_FORMAT_UNSUPPORTED。可选部件允许跳过。 |
| F02 特征顺序错误 | 从校验过 SHA-256 的 ONNX/UBJ 及其 schema sidecar 读取模型特征顺序，按 result_column 匹配请求并重排物理列。缺失/不匹配元数据拒绝执行。真实联测进一步发现导出器没保存 ONNX 元数据，已修复 Boosting wheel。 |
| F03 SQL 语义丢失 | query 必填；只接受能等价映射的单表直接投影、参数化标量过滤、简单 LIMIT。校验 AS 别名和标识符大小写，按占位符类型解析参数，禁止擅自把 String 改为数字。复杂表达式/隐式类型转换明确拒绝。 |
| F04 产物清单校验 | 校验部件/格式枚举、format 和路径唯一性、安全相对路径、size/hash、prefix 末尾斜杠。Bundle 快捷参数不能绕过 wire 清单。读取所选 alternative 的所有文件并核验大小与摘要。 |
| F05 超时/早停 | 原生 Ray Job deadline/stop + driver 阶段检查，训练 patience 接入现有 early stopping。接收时刻贯穿 admission、恢复、driver、watcher。 |
| F06 嵌套结构与实体 | 统一校验训练/推理嵌套数据、拓扑、采样、调优和标签映射；训练使用 result_column，实体跨 Ray 分区分组去重并告警。 |
| F07 推理事件 | STARTED 实测总量、Ray Queue 由 driver 汇总行计数、配置的行粒度、最终 PROGRESS=100。worker 不发布 Redis 事件。 |
| F08 取消/时长 | 校验前取消、恢复已有进度/阶段快照、接收时刻起算耗时；迟到 admission 不能覆盖运行阶段。 |
| F09 错误码 | 结构、模型缺失/损坏/不兼容、输入查询/缺列、预测、结果写入、训练各阶段、超时均使用协议码及安全消息。 |
| F10 凭证引用 | credential_ref 优先，引用缺失明确拒绝，不退回残留 inline password。 |
| F11 扩展与 AUTO | 容忍不认识或无法解析的 extensions；合法 AUTO 降级 MANUAL 并 warning，非法 tuning 先拒绝。 |
| F12 JSON | Decimal 精确转字符串，递归处理 numpy/日期/容器，NaN/Inf 转 null。 |
| F13 UTF-8 | 单消息解码故障隔离；无效载荷终态和 ACK 后继续消费。 |

独立复核发现的多分类重复编码、缺省标签映射、SQL 大小写问题均已修复并复验。真实多分类 UBJ 缺省 label_mapping 时从产物取得类别数。

## 验证记录

- 全部 KnoVa 回归：430 passed，测试进程退出码 0；JUnit 见 `remediation-regression.xml`。
- Core Broker/CLI/Ray Job：21 passed。
- Broker unit/driver transport：90 passed，8 项仍因既有 `tributo.training.xgboost_trainer` 已删除而失败；KnoVa 使用独立 operation preparer，不走该旧接口。原基线为 86 passed / 9 failed。
- 真实临时 Redis：8/8 协议探针通过，包含非法 UTF-8、取消优先、取消快照、终态互斥、ACK/pending。
- 原生 Ray 跨三个分区实体去重：通过，3 行输入得到 2 个唯一实体，一条告警；证据 `native-ray-entity-grain.json`，脚本 `scripts/verify_training_entity_grain.py`。
- Boosting 导出：11 个真实二分类/多分类/回归模型训练→导出→ONNX Runtime 数值比对通过，另有 wheel RECORD 校验；证据 `boosting-onnx-schema-repair.json`。
- 独立二审复验：推理/协议 141 passed，三项复核问题关闭。
- 本地容器联测：待最终记录。首轮真实测试发现 ONNX 导出缺少 schema，并据此修复，没有把失败首轮记为成功。

## 交付和能力边界

本次改动位于 KnoVa、Broker 和 KnoVa 自带 Boosting wheel；Core 没有新增执行框架。Broker 与 Boosting 都保留可重放的 vendor 补丁和 wheel 校验记录。测试镜像仅在本机构建，没有推送或部署生产。

当前支持数值特征的 XGBoost、单表直接查询、ClickHouse、NAS/NFS/S3。复杂 JOIN、时序 pivot、分类编码/标准化链及 stratify 尚不在当前执行能力内；这些路径会明确拒绝，不能据此声称覆盖 v2 所有算法/业务形态。没有 feature_names 的历史 ONNX 需重新导出，不能再凭请求数组猜测顺序。

写入仍遵循 APPEND 非事务语义；取消/失败不会回滚已写入的数据。进行中的 writer 没有逐批提交回调，失败/取消的 result_rows 是最后确认的提交快照，可能低于远端部分写入后的最终实际行数。长训练轮次的强制超时/取消依赖 Consumer 的原生 Ray Job stop，阶段边界另有 driver 协作检查。

生产 Redis ACL、生产者 Stream 设置、Sentinel 故障切换、生产后端完整路由不在这次本地修复复验范围内。


## 复跑命令

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=src:../tributo-broker-redis/src:../tributo/src \
.venv/bin/python -m pytest tests -q

# 一次性的隔离 Docker Compose 网络，无主机服务端口映射。
ACCEPT_IMAGE=knova-tributo-devbox:v2.8-protocol \
ACCEPT_ARTIFACTS=/absolute/path/to/evidence ACCEPT_KEEP=1 \
bash scripts/acceptance_local.sh

# 复用上面的隔离模型和结果表，使用独立 Redis namespace。
ACCEPT_IMAGE=knova-tributo-devbox:v2.8-protocol \
ACCEPT_ARTIFACTS=/absolute/path/to/evidence \
docker compose -p knova-local-accept -f scripts/acceptance.compose.yaml \
run --rm --no-deps --entrypoint python workloads /scripts/acceptance_protocol_extra.py
```

本地 macOS 一次无 OpenMP 线程约束的运行在打印全量通过后出现 native 库退出错误；上述限制线程的复跑干净退出。容器内 Linux 真实业务链路单独验证。
