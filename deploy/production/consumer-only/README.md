# 复用既有生产组件：Consumer 一键部署与分层自查

镜像：`knova-tributo-devbox:v2.7`。部署包：`knova-consumer-only-v2.7.tar.gz`。
v2.7 的本地 Kind 核心部署、常量特征正类 0 训练及 SHAP 推理已验证；完整业务矩阵另见主验收报告。
v2.6 因模型导出缺陷不再作为最终交付。

目标环境已具备 **Redis Sentinel、ClickHouse、Ray Head/Worker、对象存储**。
本目录只部署 Consumer；Tributo Core、KnoVa、Redis Broker、XGBoost 适配器在同一镜像中，
一个 Consumer 同时处理训练与批量推理。默认验收路径为 **XGBoost + ONNX + TreeSHAP**。

- `deploy.py`：只依赖运维机的 Python 3.10+ 和 kubectl。
- `doctor.py`：在临时、不消费任务的 Pod 内执行；通过 Ray 原生 Jobs API 分发代码，
  Driver 自查后，用原生 NodeAffinity 在每个存活节点执行相同检查。
- **稳态资源只有一个不可变 ConfigMap 和一个 Consumer Deployment**。
  `check/apply/rollback` 会暂时创建 doctor Pod，结束后删除；只保留安全 JSON 报告与 ConfigMap。
- 不创建/修改 RayCluster、Redis、CK、S3、Service、Secret、PVC，不修改现有 Ray Pod。
- 父目录的 `deploy.py` 会新建 RayCluster，不适用于此处的已有集群接入。

旧的 `consumer.yaml` / `kustomization.yaml` 保留为手工参考。
生产上线请用下列脚本流程，直接 `kubectl apply -k` **会绕过预检与回滚保护**。

## 一、填写三个文件

```bash
cp settings.example.json settings.json
cp config.example.json config.json
cp probe.example.json probe.json
```

### settings.json：部署位置

必须填写 `context`、`namespace`、`name`、`image`、`secret`。所有 kubectl 调用显式携带 context，
不依赖“当前 context”。namespace 与 Secret 需运维预先准备。镜像应使用不可变标签或 digest。

- `architecture`：Consumer 的 `arm64` 或 `amd64`，必须与镜像一致。
- `config`、`probe`、`output`：相对 settings.json 的路径。每个集群/namespace/name 使用独立 output。
- `old_deployment`：迁移其他旧 Consumer 时填写名称；首次安装或原地更新省略。
- `timeout_seconds`：整段自查/rollout 的超时，建议 360～600 秒。
- 可选 `node_selector`、`tolerations`、`image_pull_secrets`（名称数组）、`service_account`、
  `dns_policy`、`dns_config`、`resources`。
- 可选 `volumes` / `volume_mounts`：适用于已有凭据文件、CA 或共享文件系统。卷中的 Secret/PVC 必须已存在。

运维 kubeconfig 需要目标 namespace 中 ConfigMap/Pod/Deployment 的 get/list/create/patch/delete、
Pod logs、Deployment scale 权限以及 namespace/Secret 的 get 权限。脚本不输出或保存 Secret 的 data。

### config.json：真实业务连接

| 字段 | 检查方法 |
| --- | --- |
| `transport.mode` | 固定 sentinel |
| `sentinel_nodes` | Consumer 可达的 Sentinel 节点，通常 26379 |
| `driver_sentinel_nodes` | Ray Pod 可达的 Sentinel 节点；不是数据节点地址 |
| `sentinel_master` | Sentinel 中实际 master 名称，不是 Kubernetes Service 名称 |
| `database` | 与 ai-model-control 一致，常用 3 |
| `password_ref` | Redis 数据节点密码，如 `env:KNOVA_REDIS_PASSWORD` |
| `sentinel_password_ref` | Sentinel 自身密码；无认证时移除此项 |
| training/batch_inference stream、group | 与上游、旧 Consumer 完全一致；不要靠新建 group 跳过故障 |
| `group_start_id` | 新组使用 0-0 会处理历史任务；已有组不受此参数影响 |
| `execution.ray_dashboard_url` | Ray Jobs HTTP API，通常 Head Service 的 8265 端口 |
| `project_root` / `extra_py_modules` | Consumer 内实际分发源码路径；可以是现场自定义布局 |
| `max_driver_input_bytes` | 默认 262144，即 256 KiB；不是数据行数上限 |

**不覆盖 PATH/PYTHONPATH，不把凭据正文放到 execution.env_vars。**
Ray Head/Worker 必须自己能读取凭据引用，Consumer Secret 不会自动传给它们。
py_modules 分发 Python 项目源码，不会自动安装 C++ 动态库或所有第三方依赖。
本镜像基线 Python 3.13 / Ray 2.55.1；doctor 对所选用途的依赖版本及实际源码哈希逐一比较，
允许目录不同，也不要求 Ray 镜像存在 `/opt/tributo-runtime/verify_image.py`。

### probe.json：检查什么数据源和用途

默认 `algorithms=["xgboost"]`、`inference_formats=["onnx"]`，检查 XGBoost 与 ONNX 导出/执行依赖。
`min_alive_nodes` / `min_total_cpus` 是可选下限；不要求固定集群规模。
`required_env` 只填写变量名，不填写值。它会在 Consumer、Driver、各节点分别检查。

- `require_idle=true`：拒绝仍有 PENDING/RUNNING 的 `tributo-*` Ray Job。`apply` 必须开启。
  日常在线 `check` 可设 false，但部署前必须恢复 true。
- Redis：分别检查 Sentinel 认证/发现 master、Redis 数据节点认证/DB/连接。
  默认 `redis_stream_permissions=true`，在随机 `tributo-doctor:<UUID>` 上执行
  XADD/XGROUP CREATE/XREADGROUP/XACK/DEL 并清理，**不会消费真实任务**。
  此结果仅证明临时前缀上的命令权限；还需核对生产 task/event/cancel key 的 ACL。
  若现场禁止临时前缀，可显式关闭，此项输出 `NOT_CHECKED`，不能解释成 ACL 已验证。
- ClickHouse：按 `clickhouse` 中的显式 datasource/凭据引用检查 SELECT 1、输入/输出表 LIMIT 0。
  26.3 实测支持只读 `CHECK GRANT INSERT ON db.table`，因此可定位输出表缺少 INSERT 授权；
  版本低于 24.11 标为 `NOT_CHECKED`。这不证明结果表的列类型/DateTime64 时区符合业务。
- S3：在 `tributo-doctor/<UUID>` 执行 HEAD bucket、PUT、GET、DELETE，所有节点必须通过。
  不修改真实模型对象。检查会对照真实模型/SHAP 读取所用的 boto3 默认凭据链与 endpoint；
  仅让 probe 显式引用另一套可用凭据不算通过。Ray Pod 的 AWS_* / IAM、AWS_ENDPOINT_URL
  必须与实际任务配置相符。若 DELETE 失败会报告 probe_key，运维需清理该临时对象。
- 缺少 S3/CK probe 时明确输出 `NOT_CHECKED`；需要现场填写真实任务数据源，不能用一个示例源代表所有租户。
- 共享文件系统/NFS 场景需自行确认所有 Ray 节点相同路径内容共享、可写；本脚本 S3 检查不能代替 NFS 检查。

## 二、导入镜像、预览、自查、部署

将独立交付的 `knova-tributo-devbox-arm64-v2.7.tar.zst` 与同名 `.sha256` 一起放到服务器。
以下命令只导入当前节点，镜像架构必须匹配 Consumer 节点：

```bash
bash import-image.sh knova-tributo-devbox-arm64-v2.7.tar.zst containerd
# Docker 环境把 containerd 换成 docker；nerdctl 环境换成 nerdctl。
```

每个可能调度 Consumer 的节点均须导入，或确保私有仓库可拉取。
默认 `imagePullPolicy=IfNotPresent`，生产用不可变 tag/digest，避免复用标签造成旧镜像缓存。
既有 Ray 节点无需强制替换成 Consumer 镜像；以 doctor 的依赖、源码与凭据检查结果为准。


```bash
# 只渲染 JSON 清单，不联系 Kubernetes、不消费任何任务。
python3 deploy.py --settings settings.json render > rendered.json

# 创建临时 doctor Pod，输出每层结果；失败不会停旧 Consumer。
python3 deploy.py --settings settings.json check

# 运维先暂停上游的新任务投递。--upstream-paused 表示这一步已经完成。
python3 deploy.py --settings settings.json apply --upstream-paused

python3 deploy.py --settings settings.json status
```

上线顺序：

1. 校验本地配置与所有权，检查 Consumer 环境、Redis、Ray Driver 与全部存活节点、所选 CK/S3。
2. 检查没有活动 Tributo Ray Job，保存原 Deployment 清单、UID、副本数到 `output/state.json`（0600）。
3. 缩容旧 Consumer 并等其 Pod 完全退出；再次检查活动 Ray Job，避免旧 Pod 在预检期间接收新任务。
4. 启动新 Consumer，等待 readiness 成功。失败时先保存安全 Pod 诊断，再停止新 Consumer。
5. 回滚前确认没有活动 Tributo Ray Job，随后恢复旧清单和副本数。
   如果 Ray 不可达或仍有活动作业，**保持暂停并保留 journal**，不冒险同时启动两套 Consumer。

相同清单再次 `apply` 会校验对象 UID、spec 与就绪状态后返回 `deploy.unchanged`，
不重启 Pod、不覆盖最初的回滚快照。相同名字对象被删除重建、所有权改变、spec 被手动改动会拒绝覆盖。
升级到不同清单会把上一份 state 存为 `state-previous-*.json`，新的 state 用于回滚本次升级。
只在一个运维位置执行部署；output 的文件锁防止同目录并发命令，不是跨电脑的集群分布式锁。

**预检只证明所测配置可连接和导入。恢复上游前，还要执行一条小训练及带/不带 SHAP 的推理，
确认业务终态、模型文件和真实结果表。** 脚本不冒充真实业务验收。

## 三、失败定位速查

每条结果包含 `stage/status/code/hint`；节点失败有 `node_id`，Ray 探针有 `submission_id`。
原始 exception 文本、traceback、Secret 值不会进入报告。
rollout 失败会短暂读取近期 Pod 日志，仅输出固定异常模式分类，例如 NoPermissionError → E_REDIS_ACL；
不保存/打印原始日志，未命中模式会标为 E_LOG_UNCLASSIFIED。检查记录保存在 output 下的 `*-check-*.json`。

| code / 现象 | 优先检查 |
| --- | --- |
| E_SETTINGS / E_CONFIG | 占位符、字段名称、凭据引用、PATH/PYTHONPATH、Sentinel mode |
| E_SECRET | 报错层的 Pod Secret/挂载；Consumer 成功不代表 Ray 有相同变量 |
| E_SENTINEL | Sentinel 密码、26379 网络、master 名称；单节点失联会 WARN，只要其他节点可用 |
| E_REDIS_ACL / E_REDIS_AUTH | 安全日志分类识别的 Consumer 业务 key 权限/认证异常；核对生产前缀和两层认证 |
| E_REDIS | master 发现后返回的数据节点 IP/6379 是否可达、数据密码/ACL/DB、probe key 权限 |
| E_IMPORT | 指定模块缺失/原生库缺失、Python/Ray/所选算法依赖版本不一致 |
| E_SOURCE | 运行时实际源码被旧 py_modules 遮盖；报告 expected/actual 路径和哈希 |
| E_CK | HTTP 端口/TLS、env 引用、输入/输出表、SELECT/INSERT 权限 |
| E_S3 | endpoint/bucket、凭据、Put/Get/Delete、临时对象清理 |
| E_TOPOLOGY | 活节点数/CPU 总量不足，Worker 资源声明未生效 |
| E_RAY | Jobs API、runtime_env 安装、Ray/Python 版本、Head/Worker 网络 |
| E_BUSY | 等待现有训练/推理完成，暂停上游；不要用换 group 绕过 |
| E_PREFLIGHT_TIMEOUT | doctor Pod Pending/ImagePullBackOff；看 status 的 waitingReason/conditions |
| E_JOURNAL / E_IDENTITY / E_DRIFT | 恢复快照未完成、对象同名替换或人工变更；先确认资源身份 |
| rollout.diagnostics | 查看保存的 rollout-failure.json：调度、拉镜像、OOM、CrashLoop、readiness |

Readiness 的 health 不能证明算法执行成功；Ray 全局 FutureWarning 也不等于失败原因。
实际训练日志用 Consumer 的 operation_id → submission_id → `ray job logs <submission_id>` 关联。
现有 ai-model-control 历史日志覆盖、前端状态更新/登录续期问题不属于此部署脚本修复范围。

### 当前训练参数边界

本轮业务验收采用数值特征、手动参数、固定轮数的 XGBoost 与 TreeSHAP。
`resource_limits.max_epochs` 有效；早停使用 `algorithm.hyper_params.early_stopping_rounds`，
并要求非空验证集。协议字段 `resource_limits.early_stopping_patience` 和
`max_training_time_seconds` 当前被接受但未执行，不能将它们当作有效的早停或强制超时保护。
生产任务如果必须具备总时长硬限制，需要先补齐并验证这一能力。

## 四、明确回滚

保持上游暂停：

```bash
python3 deploy.py --settings settings.json rollback --upstream-paused
python3 deploy.py --settings settings.json status
```

脚本只恢复同 UID 的旧对象。首次安装回滚会删新 Deployment；旧 ConfigMap 保留，避免破坏恢复依赖。
不要删除 output/state.json，或在不理解当前 phase 时手工扩容旧/新 Consumer。

## 五、现场上线清单

- [ ] context/namespace/image digest/CPU 架构正确，镜像已分发到可调度节点。
- [ ] Redis Sentinel 与数据节点两层密码正确；发现出的节点地址 Consumer 和 Ray 都可达。
- [ ] training 与 batch_inference 的 DB/stream/group 与 ai-model-control 一致。
- [ ] Ray Head、Worker 都有实际任务需要的 Redis/ClickHouse/S3 引用变量与 CA 文件。
- [ ] doctor 输出实际 Driver 路径/源码哈希正确，未被旧 py_modules 覆盖。
- [ ] min_alive_nodes/min_total_cpus 与训练的 num_workers/cpus_per_worker 相容。
- [ ] CK 输入字段/目标字段类型，分类类别/回归目标正确；输出表列类型、时区、INSERT 权限正确。
- [ ] S3 模型 UBJ/ONNX、feature_names、metadata 可访问；SHAP 使用同一模型版本的原生模型/特征顺序。
- [ ] 所有 NOT_CHECKED 项都有现场补充证据；多租户数据源逐个核验。
- [ ] 上游暂停；旧任务已结束；apply 成功且重复 apply 不重启 Pod。
- [ ] 小训练 + 阈值推理 + SHAP 的业务终态与落库验证完成，再恢复上游。

## 本地验证范围

单元测试覆盖配置拒绝、无密码异常、不同目录源码比对、故障阶段、预检失败不切换、成功迁移、
rollout 失败回滚、活动作业阻止恢复、重复 apply 幂等及 UID 防误覆盖。
真实本地 Kubernetes 接入与故障注入结果由本轮验收报告记录；**不把本地成功写成生产集群已验证**。
本次 v2.7 部署验收状态见 [ACCEPTANCE.md](ACCEPTANCE.md)，随包安全断言在 `evidence/`。
