# v2.5 本地综合验收

日期：2026-09-24 至 25（北京时间）。范围：本机 Docker、真实 Ray Head/Worker、Redis、ClickHouse、RustFS、KnoVa 浏览器前端。没有访问生产 Kubernetes。

## 唯一被交付的镜像

- `knova-tributo-devbox:v2.5`，`linux/arm64`。
- 镜像 ID：`sha256:9f9c8077ebe4f5971380d43b363b7c1494f04a3f39861ec0a0cf2e8974569101`。
- 本次开发中曾使用其他同名本地候选镜像，均未作为最终交付；中间候选结果不能替代本镜像的验证。
- 最终归档名称、大小和 SHA256 见 `release.json`。

## 已实测的前端链路

独立新模型：**本地全面验收-0924**，model ID `mdl-eqaamo8fmsbukwbs`。前端镜像为 `knova-webui-uat:20260921111523`，控制服务为 `ai-model-control:main-20260921105903`。

### 新训练

v5 `ver-cdi5oyw3rgfi02ck` 经浏览器配置并发起：XGBoost 二分类、六个数值特征、834 行筛选数据、65/15/20 拆分、两 Worker、10 轮、共享 S3 checkpoint。实际切分为 542/125/167，前端和后台均 COMPLETED；产物 manifest、UBJ、ONNX、feature_names 的对象及 SHA256 已核对。

Ray submission：`tributo-redis-training-7f37f9c4967c0b9c`，operation：`lp-c23109e9c4a2`，job：`job-bdn8632m0x48tscv`。前端可查看完成状态、样本拆分、混淆矩阵、ROC/PR、特征重要性和三份模型产物。AUC 为 0.9763681592，仅用于证明该本地任务产生有效指标，不代表生产模型效果。

### 六轮推理

任务 `itask-gdgw7iu4xz6c7j` 固定 v5，实体 `user_id`，默认筛选 `user_id <= 10000`。

| 场景 | 实际输入/输出 | 结果 |
| --- | --- | --- |
| 阈值 0 | 834 / 834 | 正类 834、负类 0 |
| 阈值 0.5 | 834 / 834 | 正类 4、负类 830 |
| 阈值 1 | 834 / 834 | 正类 0、负类 834 |
| 临时筛选 `user_id <= 0` | 0 / 0 | COMPLETED，没有误跑全量 |
| SHAP 开启、10% LIMIT | 83 / 83 | 每行六特征贡献齐全，均为有限值 |
| 浏览器取消 | 0 行输出 | 前端/PG CANCELLED，Ray STOPPED |

前三轮实体集合完全一致，逐实体预测概率完全相同，所有标签均符合 `p >= threshold`。SHAP 使用 exact/RAW_MARGIN，`sigmoid(base + 各特征贡献)` 与实际预测概率最大绝对差为 `6.79e-8`。

负面交互也已实测：未选择任何特征时前端阻止保存；打开当前不支持的 stratify 后训练明确 FAILED，并实时显示具体配置错误。刷新后错误说明丢失属于下节的 webui 缺陷。

## 本轮修复并复测

1. 支持前端实际发出的受限单表参数化筛选查询，按原生 typed transforms 执行过滤、LIMIT 和列选择；过滤列不会误加入训练特征。
2. 修复真实小数据空分片：按 Worker 数使用 Ray 原生 `repartition(..., shuffle=False)`，随后 `streaming_split(equal=False)` 保留全部余数。独立原生 Ray 场景和前端 v5 均确认成功。
3. consumer 周期性接管过期 pending 消息，推进 claim 游标并去重缓冲。
4. 保留安全配置错误摘要，未知异常不输出原始异常正文；Ray 日志记录实际模块路径、任务关联、阶段及脱敏调用帧。
5. 推理 OOM 重试的减半批大小优先于原始配置；不支持的 ORDER BY 明确拒绝，不静默忽略。
6. 部署切换记录可恢复、失败后能重试，回滚校验资源身份；预检沿用业务 runtime_env 并检查全部三个分发包，避免两套代码不一致。
7. 打包脚本启用 Bash `pipefail`。模拟 `docker save` 部分输出后退出 37，确认脚本同样失败且不生成校验和/平台文件；实际归档另核验了内部 Config、RepoTags 和 ARM64 平台。

## 验证范围和限制

- KnoVa 测试全集：本地源码依赖环境 **200/200**；隔离环境只安装实际交付的 vendor wheels 再跑 **200/200**。这两次覆盖同一套用例，不相加成 400 个独立用例。
- 生产部署与预检单元测试 39 项已通过，包含在上述全集中。
- Ruff 与 shell 语法检查通过。
- 最终镜像双节点原生预检通过：Head/Worker/Consumer 镜像相同；KnoVa、Redis Broker、Boosting 安装目录与分发目录一致；Driver 和 Boosting native 的实际导入路径位于 Ray `py_modules`，Redis、S3、共享目录读写成功，临时探针已清理。Job 为 `knova-deploy-check-24780ec1dd5b`，详见 [PREFLIGHT-V25.md](PREFLIGHT-V25.md)。
- 最终镜像工作负载 **17/17 通过**：30 万行 × 116 特征、三分类、0/0.5/1 推理、S3 发布、错误输入/凭据/网络、超 256 KiB 拒绝、约 200 KiB 输入抵达 Driver、提交前后取消。总耗时约 916.89 秒，其中大数据训练 91.23 秒；这是小轮数链路验收，不是生产吞吐基准。
- 最终镜像 Sentinel **8/8 通过**：鉴权/DB3、单节点不可用、两类错误密码/缺失凭据、错误 master 名称和真实主节点故障切换。
- 通用 broker v1 训练路径有明确兼容限制：当前 broker wheel 导入 `tributo.training.xgboost_trainer`，交付的 core wheel 没有该模块；完整 broker 源码测试为 86 通过、9 失败。在最终镜像内仅使用安装包直接调用该入口，也复现相同 ModuleNotFoundError。KnoVa 使用独立 operation preparer/driver，不走此路径。不能宣称通用 broker 全部兼容。
- Broker / Boosting 的基底与本地补丁已记录并重放验证。ray-clickhouse wheel 包含先前本地修改，已记录确切 wheel hash，但完整可移植补丁链和字节一致重建尚未建立；源码复现范围见仓库 `vendor/wheels/README.md`。
- 本轮未重新进行 Kubernetes 集群部署、生产网络验证或生产容量压测；同包 `ACCEPTANCE.md` / `LOGGING-RECHECK.md` 为历史 v2.4 验证，不冒充 v2.5 结果。

## 一键部署与离线交付边界

步骤与回滚命令见 [README.md](README.md)，完整镜像导入和最小增量重建见 [OFFLINE-V25.md](OFFLINE-V25.md)。生产 namespace、KubeRay CRD、ARM64 节点、Secret 和可用存储由现场提供；每个节点都需要获得同一个镜像。

填写实际服务器配置和 settings 后执行 `python3 deploy.py apply --settings settings.json`。接管旧 Consumer 时设置 `previous_consumer`，暂停上游并等待旧 Ray Job 结束。工具在预检通过后才切换 Consumer；保留发布输出目录的 `switch.json` / `manifest.json` 以便回滚。生产放量前仍需验证现场训练、推理结果与资源配额。

## 当前 webui / 控制服务待修问题

这些问题来自现有独立前端/控制服务镜像；本次没有其可构建业务源码，未修改认证数据库或前端 bundle 绕过验证。

| 优先级 | 问题 | 已定位原因与修复方向 |
| --- | --- | --- |
| P1 | 登录约十分钟后退出 | `knova-control` 的 refresh JTI 被用于 access `jwt_id` 查询，而会话关联在 `refresh_id`；修正 refresh 查询并保留撤销、过期、轮换校验。 |
| P1 | 历史事件日志缩水 | 控制服务先追加事件，再保存带旧 event_log 的实体，覆盖新日志；通用 save 排除 event_log，或在事务内原子更新状态与事件。Redis 事件和 Ray 日志仍存在。 |
| P2 | 推理结果按钮不可用 | webui 将按钮写死 disabled；“同步”仅占位提示。此次结果正确性由后台真实 ClickHouse 数据核验，不能宣称前端结果预览通过。 |
| P2 | 详情仍显示等待中 | SSE 只更新列表，不更新详情对象；另当前适配器不发 STARTED，控制服务收到首 PROGRESS 也不把持久状态置为 RUNNING，需统一协议状态转换。 |
| P2 | 阈值显示不一致 | `Number(null)` 产生虚假的“已保存 0.00”，评估网格又吸附到 0.05；应先处理空值并统一评估阈值读数。实际推理 0/0.5/1 override 已验证正确。 |
| P2 | 刷新后失败原因消失 | FAILED 版本重新打开时未读取 training-status 的 errorMessage；应补一次状态读取，不重复触发训练。 |
| P2 | 窄窗口难以操作 | 三列工作台未适配约 655px 宽度；需响应式布局。功能验收使用 1440px，已恢复原窗口。 |

另有非阻塞文案/配置差异：固定版本仍显示“使用最新版本”徽标；UI 声称按资源选项创建新集群，但此适配器使用已有 Ray 地址。训练 Worker 数、每 Worker CPU 和 GPU 开关实际读取 `extensions.tributo.training_runtime.ray`，当前 UI 资源控件没有映射到这些字段。部署前请区分 Kubernetes 集群容量、算法 Worker 资源与 UI 尚未接通的控件。

建议下一个改动先修复 **ai-model-control 保存旧实体时覆盖 event_log**，并验证完成、失败、取消后刷新仍能完整回放历史。STARTED/RUNNING 协议衔接、详情 SSE 更新和训练失败说明回填仍是独立待修项。具体执行 ID、逐项证据和源码定位见附带 `evidence/frontend/` 报告。
