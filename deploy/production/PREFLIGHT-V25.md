# v2.5 原生 Ray 预检验收

日期：2026-09-24。使用本机现有 Ray Head、Worker、Redis 和 S3，独立临时容器运行生产预检脚本；没有重启现有容器，没有连接生产服务，也没有创建新的 Kind 集群。

本轮最终候选镜像为 `knova-tributo-devbox:v2.5`，`linux/arm64`，镜像 ID 为 `sha256:9f9c8077ebe4f5971380d43b363b7c1494f04a3f39861ec0a0cf2e8974569101`。本轮已核对 Head、Worker、Consumer 三个容器均运行该镜像；确认前端业务 Job 全部终态、Ray 无 RUNNING / PENDING Job 后才启动预检。

此镜像包含训练筛选、受控错误摘要、推理排序拒绝和 Boosting 原生 repartition 修复。本报告确认最终候选镜像的依赖预检通过；真实 UI 训练、推理、SHAP、取消和完整 workload 验收分别由对应业务报告记录，不能用依赖预检替代。

## 真实服务结果

最终候选镜像 Ray submission ID：`knova-deploy-check-24780ec1dd5b`。原生 Jobs API 返回 **SUCCEEDED**，临时预检容器退出码为 **0**。

下表对应 `9f9c8077` 镜像，新证据使用 `final-9f9c-image-preflight` 前缀。历史记录保留原文件名，范围见下文。

| 检查 | 结果 |
|---|---|
| 预检容器检查 Consumer 的 Redis / Ray 连接 | 通过 |
| 预检脚本与配置组成普通 Python 包，通过 Ray `py_modules` 上传 | 通过 |
| 保留业务工作目录并以 `python -m` 启动预检模块 | 通过 |
| 从预检模块所在目录加载配置，避免误读业务工作目录同名文件 | 通过 |
| Head 和 Worker 共 2 个存活节点，均为 ARM64 | 通过 |
| 两个节点各自连接 Redis 并执行 PING | 通过 |
| 两个节点各自执行 S3 HeadBucket / Put / Get / Delete | 通过 |
| 镜像安装目录与分发目录的 KnoVa、Redis Broker、Boosting 源码逐文件一致 | 通过 |
| Driver 和 Boosting native 实际从 Ray 分发的 `py_modules` 目录加载 | 通过 |
| 共享目录跨节点读写；检查文件随后清理 | 通过 |
| S3 预检前缀下剩余对象数 | 0 |

运行环境中的 5 个源码模块分别覆盖 Core、三个业务扩展和预检包。预检保留业务 `execution.env_vars`，不把容器进程中的 Secret 环境变量复制进上传的 runtime_env。此次实际配置未启用 `runtime_pip_packages`；pip 配置转发通过本地单元测试，没有额外访问包仓库。

本轮包含每节点 Boosting native 导入路径证据及三包一致性校验。S3 预检对象和共享探针文件均已清理，凭据扫描通过。

本地部署与预检单元测试 **39 项通过**，Ruff 检查通过。覆盖自动恢复后重试、显式回滚后重新发布、原配置缺失时回滚、资源归属和 UID 校验、恢复中断阻断、S3 bucket 变更不得静默复用旧发布，以及 Boosting 缺包或镜像校验失败时阻断。

## 历史镜像边界

- `sha256:5c89cf9ad38c3434972a5d9904516ba288b195686cc406d20515f1dcec955def`：SQL / 错误摘要修复之前，Job `knova-deploy-check-33eed14ca485` 依赖预检通过；证据为 `v25-native-preflight.*`。
- `sha256:97dc0e906e8844a777ef1963f5bf562d475ce9a65486ee689e43151a366d5577`：Job `knova-deploy-check-6411dacf6eef` 依赖预检通过，但真实 UI 小数据训练出现空 shard，业务验收未通过。现场为筛选后 834 行、训练集 542 行 / 75 个 block，一个 worker 没有训练行；证据保留 `final-image-preflight` 历史文件名，不表示最终交付通过。
- Boosting 后续修复经独立复审和精确行 ID 覆盖回归，834 / 835 行均无丢失，原生 Ray 双 worker / 3 轮训练成功；本轮 `9f9c8077` 已包含修复。详细证据见 `dist/comprehensive-recheck/INDEPENDENT-REVIEW.md`。
- `ACCEPTANCE.md` 对应 **v2.4**；其中 Kind / Kubernetes 切换结果不代表 v2.5 已重新验证。

## 一键部署的边界

填写 `settings.json` 和实际服务器 `config.json` 后，入口为：

```bash
python3 deploy.py apply --settings settings.json
```

目标 namespace、KubeRay CRD、可调度 ARM64 节点及 Secret 需提前存在；各节点必须能获取同一个镜像。脚本创建专用 RayCluster，在预检成功前保持新 Consumer 为 0 副本。

替换旧 Consumer 时，先暂停上游并等待旧 Ray Job 结束，配置 `previous_consumer`；新 Consumer 就绪失败会尝试恢复旧副本。成功回滚后可用相同配置重试；恢复过程被中断时需先显式回滚。镜像、业务配置或预检选项改变时使用新的发布名和输出目录。

本次共享存储检查使用 Docker bind mount，不代表生产 RWX PVC、CSI 或 NFS 已验证。此次没有重跑 Kubernetes 服务端校验、真实 Deployment 切换或生产网络验证；历史 v2.4 的 Kind 记录保留在 `ACCEPTANCE.md`。预检成功后仍需通过 ai-model-control 提交真实训练和推理任务，确认业务终态与结果表。

最终候选镜像证据保存在源码仓库 `dist/comprehensive-recheck/preflight/`：`final-9f9c-image-preflight.log`、`final-9f9c-image-preflight.json`、`final-9f9c-image-preflight-cleanup.json`、`final-9f9c-image-preflight-secret-scan.json`。分别记录预检日志、原生 Job 状态与节点结果、对象清理检查及凭据扫描结果。凭据文件 `frontend/runtime.env` 未复制进该目录，不得加入交付包。
