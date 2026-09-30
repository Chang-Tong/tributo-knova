# ai-model-design Redis v2 协议全面复核

检查日期：2026-09-29。结论：**当前实现满足一部分 Redis 传输与受限业务路径，但不满足 ai-model-design v2 的完整协议要求；v2.7 发布镜像同样存在这些差距。**

本次只新增审计报告、探针和结果记录，没有修改生产实现、重打镜像或变更部署。

## 基准、范围与证据

- 协议库：`/Users/dongair/project/Tributo/external/ai-model-design-docs`，提交 `358ffc407439657131be5b71212fc8afa6973a73`。本次 `git ls-remote origin HEAD` 返回同一提交，核对了远端默认分支。
- 规范：`训练推理引擎任务描述协议/protocol-v2-2606/` 的公共约定、训练协议、推理协议、训练协议附录，共 2,670 行；下文分别简称 C、T、I、A。
- Core：`/Users/dongair/project/05-pr/tributo`，HEAD `b0c73ee`。
- Broker：`/Users/dongair/project/05-pr/tributo-broker-redis`，HEAD `e2973b2`，含已有未提交改动。
- KnoVa：`/Users/dongair/project/05-pr/tributo-knova`，HEAD `19e8764`，含已有未提交改动。
- 发布镜像：`knova-tributo-devbox:v2.7`，镜像 ID 前缀 `b49ce75601f0`。隔离、禁用网络的临时容器中复跑映射探针，运行后删除容器。
- SHA-256 比对：镜像内 KnoVa 的 8 个 Python 文件、Broker 的 13 个 Python 文件与当前工作区逐一一致；镜像内 Broker、Boosting Python 文件与 vendor wheel 一致。见 [package-comparison.json](package-comparison.json)。因此下述 KnoVa/Broker 问题不是“源码有问题但发行包已经修好”。

| 验证 | 结果 | 解释 |
|---|---|---|
| KnoVa 现有 tests，显式加载当前 Core/Broker/KnoVa 源码 | 318 passed | 现有覆盖通过，不能替代外部协议验收 |
| Core broker、broker CLI、Ray Jobs 测试 | 21 passed | 公共接入与提交机制通过 |
| Broker unit + driver transport，显式加载当前 Core | 86 passed / 9 failed | 9 项均触及旧 `tributo.training.xgboost_trainer` 已不存在的问题；KnoVa 自定义 operation preparer 绕过该路径，不能混称为 KnoVa 的 9 项失败 |
| 新增定向协议探针，当前源码 | 39 项：9 PASS / 30 FAIL | 针对规范边界和已识别风险设计，**不是协议覆盖率或产品成功率** |
| 同一映射探针，v2.7 镜像 | 31 项：4 PASS / 27 FAIL | 去掉 8 项独立 Redis 测试，其余结论一致 |

源码探针包含真实、独占临时 Unix socket Redis，禁用持久化、退出后终止；Ray 提交与业务数据库执行不在这些探针中。超时使用合成时钟、推理进度使用 stub 计算结果；没有把它们表述为完整分布式运行测试。此次未重新进行真实 ClickHouse/Ray 训练推理、Sentinel 主从切换或生产网络验证。

证据：[探针源码](probe.py)、[源码结果](source-probes.json)、[v2.7 镜像结果](image-v27-probes.json)。

## 需要优先修复的协议违反

### F01 · P1 · 必需预处理产物被直接忽略

I §3.2、§3.6、§6 规定：按 `artifact_name` 推导执行链；每个 `required_for_inference=true` 部件必须有可处理的 alternative，否则 `MODEL_FORMAT_UNSUPPORTED`。

`inference.py:208` 的 `_artifact_alternative()` 只取 `model_weights`；`_build_request()` 没有加载或应用 `preprocessor` 的步骤。加入一个合法、必需的 sklearn preprocessor 后，计划仍然成功构建为直接 ONNX 推理。未知部件也能被忽略。

**影响：**模型原本需要标准化/编码后的数据，却直接接收原始值；维度恰好兼容时可能静默产生错误结果。该问题不能用“不支持 sklearn 预处理”解释，因为不支持时必须显式失败。

证据：`required_preprocessor`、`unknown_artifact_key`。修复应先校验全部必需部件，未支持的链路按协议拒绝，再复用现有产物加载机制补齐能力。

### F02 · P1 · 模型输入顺序取自请求数组，未重放产物的特征顺序

I §2.2.2、§3.1–3.3 明确：`features` 声明取数契约，模型列顺序由训练产物负责，推理侧不得重新决定。

`inference.py:339` 按请求数组提取物理列名，`:801` 直接写入 `TensorInputBinding.columns`。Core `inference/kernel.py:149` 按这些列拼接矩阵；`inference/resolver.py:116` 检查张量名/类型/布局，不校验表级特征顺序。该路径不会走旧 `XGBoostONNXPredictor` 自动读取 feature_names 的逻辑。

**复现：**同一行 `x=1,z=10`，同一模型引用，仅交换 `input.features` 两项，实际 Core 张量由 `[[1,10]]` 变成 `[[10,1]]`，无错误。可选 TreeSHAP 路径有 UBJ 特征名校验，但未启用解释时不能提供保护。

证据：`artifact_feature_order`。应从已有产物的特征 schema/预处理状态恢复顺序，并校验请求覆盖；不能依靠前端永远维持数组顺序。

### F03 · P1 · 推理 SQL 重建未保持原查询语义

C §5、§9.2，I §2.2.2、§3.3-A：`input.query` 是执行与范围权威；重建查询须等价保留过滤，不能确定时回退原查询。

`inference.py:398–476` 只提取 FROM、部分 WHERE 和 LIMIT，未验证 SELECT 投影，且丢失占位符类型：

- `SELECT x * 100 AS opaque_x ...` 被接受，计划仍读取原始 `x`，变换列表为空。
- `{p:Int64}` + `params={"p":"42"}` 被转成 `FilterEq(value="42")`，类型仍为字符串；`:917` 的计数查询又按值推断成 String，实际 Ray 过滤与原 Int64 SQL 可能出现类型错误或结果差异。
- 不等号路径会把数字外形的 String 参数自行转为数字，不能保证保留 String 的字典序语义。
- 必填 `input.query` 缺失/空 SQL 时返回空 transforms，继续扫描声明的物理表，失去后端限定范围。

证据：`typed_sql_parameter`、`sql_projection_semantics`、`required_query`。**影响包括错误特征值、错误人群范围及意外全表推理。**应优先让现有 ClickHouse binding 执行原参数化查询；对不能等价映射的形态明确拒绝，不继续扩展一套不完整的正则 SQL 解释器。

### F04 · P1 · 模型产物强制校验没有闭合

T §2.4-A.6，I §3.6、§6 要求完整性、部件/格式/路径唯一性和 prefix 校验。

`inference.py:156` 会直接剥除 prefix 的 `/`，接受不带末尾 `/` 的非法结构；`:313` 将缺失 hash 转为 `expected_sha256=None`。重复 alternative、重复文件路径、未知 artifact_name 均未拒绝。`:285`、`:292` 的显式/本地 Bundle 快捷路径还先于协议 model_artifacts 校验返回。

证据：`missing_file_hash`、`missing_prefix_slash`、`duplicate_alternative`、`unknown_artifact_key`。

**边界：**Core 对自身 Bundle 的完整性检查仍然存在；本问题是外部协议清单的强制条件及权威 hash 没有完整落实，不能据此声称 Core 完全不验 hash。应先完成协议清单校验，再选取现有 Bundle/Artifact runtime。

### F05 · P1 · 超时上限及协议早停配置未执行

I §2.2.4、§8 要求超时终止并 `FAILED(EXECUTION_TIMEOUT)`；训练请求提供 `resource_limits.max_training_time_seconds`、`early_stopping_patience`。

`training.py:483` 只使用 `max_epochs` 和另一个字段 `algorithm.hyper_params.early_stopping_rounds`；没有消费协议 patience/timeout。`inference.py:844` 的 execution policy 与 `:1428` 执行流程均未落实推理 deadline。跨三仓检索未发现这些字段的执行消费。

**复现：**推理上限 1 秒，合成时钟显示执行 10 秒，仍输出 `COMPLETED(duration_seconds=10.0)`；训练 plan 没有协议 patience/timeout。发布目录 `consumer-only/release-summary.json` 也记录了训练两项字段“accepted but not executed”。

证据：`training_controls`、`inference_timeout`。应把协议字段接入现有 Ray Job 生命周期/训练回调，在终止后上报规定错误码和进度，避免引入另一套执行框架。

## 其余确定缺口

### F06 · P2 · 请求嵌套结构与关系校验不足

`protocol.py:46`、`:71` 主要验证顶层 ID；嵌套业务结构是任意 dict。后续 mapping 也未补齐 T §三、I §六的约束。

探针确认以下非法训练请求都能构建 plan：重复 feature_id、重复 result_column、未登记 origin alias、没有 PRIMARY、category/target 不匹配、sample_ratio=100、tuning.mode=INVALID。指定 entity_key 却未 SELECT 实体列仍通过；实际读取列也未加入该实体列，后续训练没有粒度断言/去重。

训练用 `origin.column_name` 对齐，不使用 SQL 的 `result_column`；`:622` 的投影检查只比较源列且强制顺序，未检查 AS 目标。这既漏掉规范要求，也会拒绝列顺序不同但按结果列可正常对齐的合法查询。

修复：在进入 LOADING_DATA 前用嵌套模型/既有校验机制完成协议验证；加载后对结果列和实体粒度做实际检查。错误按 `INVALID_PAYLOAD` / `DATA_QUERY_FAILED` 分类。

### F07 · P2 · 推理缺 STARTED、行级 PROGRESS 及末批收尾

I §5.1–5.2 规定 STARTED 带实测行数，按 `progress_report_interval_rows` 报进度，必须含末批 100%。

`reporter.py:188` 没有 STARTED 分支，显式调用会变成空 LOG；`inference.py:1449` 仅发执行前 0%，然后直接 COMPLETED。设置 interval=5、实测 20 行时，观察到 INIT 0%、PREDICT 0%、COMPLETED，没有 STARTED 和最后一条 PROGRESS=100。

**影响：**控制端不能按规定更新起始实测总量，运行中进度停滞。COMPLETED 的 100% 不能替代规范要求的末批 PROGRESS。证据：`inference_started`、`inference_final_progress`。

### F08 · P2 · 取消时的进度、阶段与耗时失真，且检查顺序不符

T §4.2 要求队列阶段在结构校验前检查取消；I §5.6 要求取消进度快照。

- Broker `runtime.py:456` 先解析请求，`:507` 后检查 cancel；已取消的坏 JSON 返回 FAILED(INVALID_JSON)，不是 CANCELLED。
- `cancellation.py:192` 为取消新建 reporter，只传 reason。真实 Redis 中，已有 `8 processed / 4 result / 10 total / WRITE_RESULT`，取消事件变为 `0 / 0 / 0 / PREDICT`，duration=0。
- queued inference 的 `phase="QUEUED"` 原样透传，超出 I §5.0 的阶段枚举。
- 耗时由每个 reporter 构造时起算，正常 driver 完成也不包括 Consumer 接收至 driver 启动的耗时；watcher 新建 reporter 的终态尤其接近零。
- 运行中取消通过 Consumer watcher 调用原生 Ray Jobs stop，driver 不做协议描述的批/轮次边界协作检查。Ray 原生停止机制本身应保留，但需要完善生命周期上下文、收尾与协议事件。

证据：`cancel_before_schema_validation`、`cancel_progress_snapshot`、`queued_cancel_phase`。不能把已写结果行数报告为零；协议规定取消后已 APPEND 结果保留。

### F09 · P2 · 错误分类与失败阶段未映射成协议语义

T §2.6、I §5.5 定义可操作的错误码与发生阶段。

`execution_driver.py:105` 只识别推理 EMPTY_INPUT/配置 INVALID_PAYLOAD，其余统一 `EXECUTION_FAILED`。训练的 `_TrainingConfigurationError` 也如此。解析层使用 INVALID_REQUEST/INVALID_JSON/IDENTITY_MISMATCH；watcher 使用 RAY_JOB_FAILED 等内部码。接收端允许未知码回退 UNKNOWN，但这会丢失协议要求的失败分类。

`inference.py:1491` 将 Core failure 的 code/phase 压成一段异常文本；driver 的通用异常路径固定报 EXECUTING，无法区分 LOAD_MODEL、READ_INPUT、WRITE_RESULT。例如产物损坏、缺列、写库失败应保留对应结构化码。证据：`training_invalid_error_code`，以及上述源码路径。

### F10 · P2 · credential_ref 优先级相反

C §6：两者同时存在时 credential_ref 优先。`training.py:813`、`inference.py:494` 都只在 password 为 None 时解析 reference。探针用两种虚构凭证，实际取 inline-fixture 而非 reference-fixture。

该问题不等于发现真实凭证泄露；它意味着凭证轮换、残留旧 inline 值时会连错或认证失败。修复应保持已有 env 引用机制并改变选择顺序。证据：`credential_ref_precedence`。

### F11 · P2 · extensions 容忍规则、AUTO 降级告警没有完整落实

C §8 要求容忍任意 extensions 内容，不因解析 extensions 失败拒绝任务；`training.py:560` 对 `extensions.tributo` 等层级作强制 dict 校验，`{"tributo":"future-opaque-content"}` 会拒绝整个任务。

T §1.2.9-A、A §6.7 要求未实现 AUTO 时降级 MANUAL 并 warning；当前完全不消费 tuning，没有合法性校验，也没有降级告警，`tuning_result` 恒 null。**不要求首版实现完整搜索，但必须执行规定的兜底行为。**证据：`extension_tolerance`、`invalid_tuning_mode` 和训练调用链。

### F12 · P2 · Decimal 原始特征不能写入 pred_extra

I §4.5 明确 Decimal/numpy 转 JSON 原生值，NaN/Inf 转 null。`inference.py:971` 处理 numpy、NaN/Inf、datetime，但原样返回 Decimal；`:1230` 附近 json.dumps 会抛 TypeError。合法 Decimal 特征可能在预测结束后的结果投影阶段失败。证据：`decimal_json_safety`。

### F13 · P2 · 非法 UTF-8 消息可终止 Consumer 循环

`redis_client.py` 启用 decode_responses；`consumer.py:93` poll 只捕获 TimeoutError。向隔离 Redis 写入 `payload=b"\xff"` 后，`runtime.run_once()` 抛 UnicodeDecodeError，不能生成协议失败事件或 ACK；CLI 没有逐消息异常隔离。

这是坏消息健壮性缺口，区别于正常协议任务能力。应将传输解码失败限定在单条 delivery，保留外层可恢复 ID 并按无效载荷处理。证据：`invalid_utf8_contained`。本次未把 Python NaN 当作“必然让 Consumer 崩溃”的结论；其 JSON 归一化路径与 UTF-8 解码不同。

## 支持范围与逐项结论

不同引擎可以按算法注册表路由，不要求每个 Ray 引擎实现协议里的所有算法/存储/框架。下面的“范围限制”本身不全部构成缺陷；但必须对后端公布能力，并对不支持的必需操作明确失败，不能声称覆盖完整协议。

| 领域/条款 | 状态 | 证据或限制 |
|---|---|---|
| task entry = identity + payload | 通过 | consumer 精确校验两个外层字段 |
| event entry 非扁平化、ID 一致 | 通过 | reporter 使用固定 Map，JSON 内同一 ID；现有测试覆盖 |
| protocol_version 主版本兼容 | 通过 | 2.9 接受、未知字段不致拒绝；3.0/缺版本有结构化拒绝 |
| 不兼容版本 ACK | 通过 | 真实 Redis FAILED+pending=0 |
| timestamp epoch 毫秒、日志小写 | 通过 | reporter 统一生成 |
| event MAXLEN=1000 默认 | 通过 | 配置默认；真实 Redis 测得指定 5 时长度为 5 |
| task MAXLEN、Redis ACL | 外部待核 | 属于生产者/部署配置；本次未连接生产 Redis，不能替控制端确认 |
| 终态互斥、终态后禁止写事件 | 通过本次实测 | 24 并发终态 + late LOG，只写入 1 条；Lua 实现依赖现有终态留在 Stream 中 |
| 单写者、阶段顺序 | 部分 | worker 不直接写 Redis；Consumer admission、driver、watcher 是多个进程写者。Lua 保终态互斥，但不能保证 admission 事件不会晚于 driver PHASE |
| pending 恢复 | 通过本次实测 | XAUTOCLAIM 成功取回 staged delivery；不等于跨服务故障全链路验证 |
| native Ray Jobs 幂等提交 | 现有单元通过 | Core 相关 21 项通过；不承诺跨历史清理的 exactly-once |
| cancel key 存在即取消、空值有效 | 通过 | 真实 Redis CANCELLED+ACK |
| 取消快照/协作边界/校验前检查 | 不符合 | F08 |
| 接收端完整结构/拓扑校验 | 不符合 | F06 |
| features.result_column 不透明对齐 | 不完整 | 实现使用物理列重建且忽略部分别名关系 |
| query + typed params 等价执行 | 不符合 | F03；未执行 resolved_sql 这一点符合规范 |
| 单表、DIRECT_QUERY、数值 XGBoost | 受限路径支持 | baseline mapping、现有单测及历史验收记录；本次未重跑真实业务 |
| 多表 JOIN、TIMESERIES_PIVOT | 范围限制 | 当前映射拒绝；合法复杂 SQL 超出支持范围 |
| CATEGORICAL/ONE_HOT/标准化等 | 范围限制 | 训练明确拒绝；不能执行协议完整训练示例 |
| stratify=true | 范围限制/验收差距 | 协议分类示例和不均衡数据要求该能力，当前直接拒绝 |
| split 比例和互斥 | 部分通过 | 比例容差、TIME_ORDERED+stratify 互斥存在；不支持分层 |
| class_amounts 与采样 | 部分 | 仅特定单表 UNION ALL shape；无加载后 >5% 偏差告警，sample_ratio/sample_limit 结构校验未齐 |
| SMOTE | 规范预留 | 不单独计为必须实现的缺陷 |
| AUTO tuning | 不符合兜底约定 | F11；无需首版实现完整 AUTO |
| 训练 max_epochs | 通过代码核对 | min(n_estimators,max_epochs) |
| timeout / protocol patience | 不符合 | F05 |
| 训练 METRICS 形状/频率 | 基本具备 | Boosting worker 汇总至 driver，指标归一、ceil(rounds/100)、首尾轮处理；未重跑长训练 |
| 终态 test-set metrics / confusion / importance | 已有路径具备 | 结果映射与现有测试；本次未重算真实数据统计 |
| artifact_manifest 训练投影 | 已有路径具备 | ONNX/xgboost、hash/size、共享 storage；问题主要在消费端 F01/F04 |
| inference 必需部件/完整性/schema | 不符合 | F01、F02、F04 |
| 推理 STARTED / PROGRESS | 不符合 | F07 |
| error_code / phase / duration | 不符合 | F08、F09 |
| 结果 APPEND 与固定结果列 | 已有实现 | 通过 WriteGateway append；没有修改生产数据验证 |
| label_mapping、正类索引 | 已有修复覆盖 | 二分类允许 positive index=0；多分类反解码；现有测试和 v2.7 wheel 补丁 |
| result_filter 原始标签值域 | 通过现有测试 | 分类 INCLUDE/EXCLUDE；回归拒绝非空 filter |
| pred_extra JSON 安全 | 部分 | NaN/Inf/numpy/datetime 已处理，Decimal 缺失 |
| credential_ref | 不符合优先级 | F10 |
| 凭证日志脱敏 | 部分验证 | 已有安全异常包装和测试；未对第三方完整日志链作渗透验证，不能给绝对无泄露结论 |
| Sentinel | 未重做运行验证 | 源码复用 redis-py；此次不是 Sentinel 故障切换验收 |
| Stream key / group 与后端一致 | 需部署联验 | 当前生产模板用 distributed 前缀；规范允许按引擎拆分。不能仅凭名称不同判协议错误 |

协议文本自身有一处需统一：C §4 训练事件写作 `knova:aimodel:training:events:{job_id}`，而 T §二及 C 的 CLI 示例用 `knova:training:events:{job_id}`；T 取消默认也用后者。当前代码可配置，建议以后端和引擎共同配置为准并消除文档歧义。不是要求硬改回某一个前缀。

## 修复顺序与复验边界

1. 先封住会静默改变结果的入口：必需预处理链、模型列顺序、SQL 等价性、hash/清单强校验。暂不支持的形态明确拒绝。
2. 接入 timeout/early stopping，并补齐结构校验、规定错误码、STARTED/PROGRESS、取消快照和统一起算时间；复用现有 Ray Job 与 driver 生命周期。
3. 修正 credential_ref、extensions、Decimal 与坏消息隔离；将本报告有效反例转成以外部规范为 oracle 的回归测试。
4. 明确并发布当前引擎能力范围，再用后端真实生成的请求对新镜像做 ClickHouse/Ray/Redis 联验，包含 string/numeric 参数、特征重排、必需预处理、取消已写入、超时和故障恢复。验收通过前不要标注“完整支持 ai-model-design v2”。

## 复跑

从 KnoVa 仓库执行，使用已有环境，显式锁定本次检查的三份源码：

```bash
PYTHONPATH=/Users/dongair/project/05-pr/tributo-knova/src:/Users/dongair/project/05-pr/tributo-broker-redis/src:/Users/dongair/project/05-pr/tributo/src \
  .venv/bin/python docs/audits/2026-09-29-ai-model-design/probe.py \
  --output docs/audits/2026-09-29-ai-model-design/source-probes.json
```

镜像映射探针（无网络、不启动其 Consumer）：

```bash
docker run --rm --network none \
  -v /Users/dongair/project/05-pr/tributo-knova/docs/audits/2026-09-29-ai-model-design/probe.py:/tmp/probe.py:ro \
  --entrypoint /opt/tributo-knova/bin/python \
  knova-tributo-devbox:v2.7 /tmp/probe.py --no-redis
```

探针输出里的 FAIL 是“当前观察未满足该项协议预期”；部分为支持范围检查，部分为真正错误，按本报告逐项区分。它们不是 30 个互不相关的生产故障，也不是 39 项就覆盖了全部协议。
