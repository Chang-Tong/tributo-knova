# KnoVa v2.5 本地验收与 Kubernetes 部署

**生产已有 Redis 哨兵、ClickHouse、Ray 时，请使用 [仅部署 Consumer](consumer-only/README.md)。**
Tributo 已包含在 Consumer 镜像中。下文 `deploy.py` 会创建独立 RayCluster，属于另一种部署方式，
不用于本次接入已有生产 Ray 的需求。专用接入包为 `knova-consumer-only-v2.5.tar.gz`。

镜像：`knova-tributo-devbox:v2.5`，平台：`linux/arm64`，Ray：`2.55.1`。
部署工具需要 Python 3、kubectl、已安装的 KubeRay，以及有权限的 kubeconfig。
历史 v2.4 的本地 Kubernetes 实测使用 KubeRay 1.6.0；脚本等待 RayClusterProvisioned 条件。
v2.5 新预检已在真实本地 Ray Head/Worker 上通过，详见 [v2.5 预检验收](PREFLIGHT-V25.md)。本轮未重新创建 Kind，生产 Kubernetes 的资源调度与切换仍需现场验证。
最终候选镜像 ID 为 `sha256:9f9c8077ebe4f5971380d43b363b7c1494f04a3f39861ec0a0cf2e8974569101`。完整镜像导出与基于精确 v2.4 基础镜像的最小增量重建命令见 [离线交付说明](OFFLINE-V25.md)。
[v2.5 综合验收](COMPREHENSIVE-V25.md) 汇总真实 UI、测试结果和已知限制；其中进行中的项目应等待完成结果。`ACCEPTANCE.md` 和 `LOGGING-RECHECK.md` 保留历史 v2.4 事实。
本工具创建一个独立 RayCluster（Head + Worker）和 Consumer，使用 Ray 原生能力。
不会修改现有共享 RayCluster。现有 DNS、调度、GPU、网络策略等自定义设置需要在生成的清单中审核。

本交付支持 KnoVa 自定义适配器任务链路。镜像内 Redis Broker 默认 `tributo-generic-v1` 训练入口仍引用当前 Core 已移除的模块，不能作为已兼容的通用训练入口使用；具体证据见综合验收的兼容性限制。

## 1. 一次性准备

1. 所有可能调度到的 ARM64 节点都要能拉取镜像。建议上传内部仓库，使用不可变版本标签或 digest。
2. 离线方式需要在**每个 Kubernetes 节点**按实际容器运行时导入；工作站 Docker 的缓存不等于节点 containerd 缓存。
3. 目标 namespace 中准备 `tributo-knova-env` Secret，Head、Worker、Consumer 会同时引用它。
4. 从实际服务器复制 `config.json`，核对 Redis/Sentinel、database、两个 channel 的 stream/group。保持生产消费者组一致。
5. 使用 NAS/NFS 时准备已 Bound 的 ReadWriteMany PVC。S3 方式可直接把 Ray checkpoint 写到 S3；请求中不要配置一个未共享的本地 `ray_storage_path`。

离线导入（同目录带 `.sha256` 文件）：

```bash
bash import-image.sh knova-tributo-devbox-arm64-v2.5.tar.zst containerd
# Docker 节点用 docker；使用 nerdctl 的节点可用 nerdctl。
```

Secret 可包含多个键。把真实值放在受保护的本地文件中，通过 `--from-env-file` 创建；不要把文件提交到 Git。

```text
KNOVA_REDIS_PASSWORD=实际 Redis 密码
KNOVA_REDIS_SENTINEL_PASSWORD=实际 Sentinel 密码（Sentinel 无认证则配置中移除对应 ref）
AWS_ACCESS_KEY_ID=实际 S3 access key
AWS_SECRET_ACCESS_KEY=实际 S3 secret key
AWS_ENDPOINT_URL=http://实际对象存储地址:9000
AWS_REGION=us-east-1
```

这些敏感值通过 Pod 的 `envFrom.secretRef` 注入，不能放进 `execution.env_vars`。
`execution.env_vars` 可保留 PATH、日志级别等非敏感选项。
生产中避免使用 `latest` 或覆盖已有版本标签；离线包校验文件只校验镜像包，不会校验另行修改的服务器配置。

## 2. 一个命令部署

解压部署包后进入 `knova-production-v2.5` 目录。把 `settings.example.json` 复制为 `settings.json`，把服务器配置保存为 `config.json`（首次部署可参考 `config.example.json`），填好实际配置后：

```bash
python3 deploy.py apply --settings settings.json
```

镜像离线导入后，settings 的 `image` 可填写 `knova-tributo-devbox:v2.5`；通过内网仓库分发时填写完整仓库地址。不用 S3 时删除 settings 中的 `s3_bucket`；使用时必须替换 `YOUR_BUCKET`。

首次部署使用这个例子即可；接管旧 Consumer 时在 settings 中增加 `previous_consumer`。
配置文件和输出目录的相对路径以 settings 文件所在目录为基准。

先暂停上游提交新任务，等待旧 Ray 集群的运行任务结束。下面参数按实际环境填写：

```bash
python3 deploy.py apply \
  --context YOUR_CONTEXT \
  --namespace lakehouse-system \
  --config ./config.json \
  --secret tributo-knova-env \
  --name tributo-knova-v25 \
  --image YOUR_REGISTRY/lakehouse/knova-tributo-devbox:v2.5 \
  --workers 1 --cpus 4 --memory 8Gi \
  --s3-bucket YOUR_BUCKET \
  --previous-consumer tributo-knova-consumer \
  --output ./release-v25
```

首次部署省略 `--previous-consumer`。如果旧 Consumer 的配置路径不同，增加 `--previous-config /实际/config.json`。
使用私有仓库认证时加 `--image-pull-secret SECRET_NAME`。
使用共享文件系统时加 `--shared-pvc YOUR_RWX_PVC --shared-path /shared/tributo`，任务的存储路径需与挂载路径一致。
`--memory` 为每个 Ray Pod 的 requests/limits，默认 8Gi；需要足够的可调度资源。
单次训练读取 `extensions.tributo.training_runtime.ray` 的 `num_workers` / `cpus_per_worker` / `use_gpu`；当前 UI 资源控件尚未映射到这些字段。部署参数控制集群容量，不代表 UI 所选任务资源已生效。

执行顺序：

1. 检查 context/namespace、ARM64 Ready 节点、KubeRay CRD、Secret 必需键和配置中的凭据误放问题。
2. 用 Kubernetes 服务端校验清单；保存 `manifest.json`。
3. 创建 RayCluster，Consumer 暂时保持 0 副本。
4. 等待 Head/Worker Ready，使用与业务提交相同的非敏感环境变量、源码模块和运行时依赖，通过 Ray Jobs 原生接口运行预检：Consumer 的 Redis/Ray 连接、每个 Ray 节点的 Redis 连接，以及 KnoVa、Redis Broker、Boosting 三个包在镜像安装目录和源码分发目录的逐文件一致性；节点结果同时记录 Driver 和 Boosting native 的实际导入路径。选择 S3/PVC 后检查读写及跨节点共享。
5. 检查旧集群无 PENDING/RUNNING Job，然后暂停旧 Consumer、启动新 Consumer。新 Consumer 未就绪时恢复旧副本数。
6. 保存预检日志和切换记录。再次执行相同版本与配置不会重新部署。

`apply` 的成功说明依赖和部署已就绪。正式放量前仍需从 ai-model-control 提交一条真实训练及推理任务，确认 COMPLETED 和结果表；该工具不会伪造业务成功事件。
本地验收结果见同包的验收报告，生产网络、实际数据和资源配额需要现场确认。

## 3. 预览、检查与回滚

把同一命令的 `apply` 换成：

- `render`：只在本地生成 JSON 清单。
- `check`：检查集群和 Secret、执行服务端 dry-run，不创建工作负载。
- `status`：显示本次发布的 Pod/RayCluster/Deployment/Job。
- `rollback`：读取同一 `--output` 下的 `switch.json`，停新 Consumer、恢复旧副本数；保留新 Ray 资源用于排障。

新 Consumer 未就绪时，脚本等待新 Consumer 停止并确认旧副本恢复，将切换记录标记为已回滚。修复原因后可直接使用相同配置、发布名和输出目录再次执行 `apply`；显式 `rollback` 成功后也可这样重试。切换过程被中断或恢复失败时，先执行 `rollback`，脚本不会把未完成的恢复当成安全重试。

回滚不再读取原 `config.json`，可以只提供保存记录的位置和发布身份：

```bash
python3 deploy.py rollback \
  --context YOUR_CONTEXT --namespace lakehouse-system \
  --name tributo-knova-v25 --output ./release-v25
```

回滚会校验新 Consumer 的资源归属、发布摘要及记录中的资源 UID；同名资源被替换时拒绝切换。请保留输出目录的 `switch.json` 和 `manifest.json`，旧版切换记录需要后者恢复发布摘要。

修改镜像、配置或 S3 检查 bucket 等预检选项时使用新的 `--name` 和 `--output`。发布摘要同时覆盖工作负载和预检 Job，已有发布不会跳过新增加的检查。工具拒绝覆盖非本工具创建的同名资源。
切换及回滚都应暂停上游并等待当前任务结束；回滚 Consumer 不会自动撤销已经提交到 Ray 的任务或已经写入的业务数据。

## 4. 排障入口

```bash
kubectl --context YOUR_CONTEXT -n lakehouse-system logs deploy/tributo-knova-v25-consumer
kubectl --context YOUR_CONTEXT -n lakehouse-system logs job/tributo-knova-v25-check
```

Consumer 日志稳定记录 `operation_id` 和 `submission_id`；内部 `ray_job_id` 在查询时已产生才会记录，提交初期或快速失败时可能为 `<unknown>`。排障使用 `submission_id`，无需等待内部 Job ID。在新 Ray Head 中：

```bash
ray job status --address http://127.0.0.1:8265 SUBMISSION_ID
ray job logs --address http://127.0.0.1:8265 SUBMISSION_ID
```

- `logs` 为空时务必看 `status`：JobSupervisor 被内存保护终止、runtime_env 失败时，Driver 可能根本没有启动。
- Head 的 Dashboard 和 GCS 本身占内存。本地已复现 3Gi Head 导致 JobSupervisor 被杀；不要把这当作算法异常，也不要关闭 Ray 内存保护来掩盖资源不足。
- Head 的运行环境代理使用 Ray 原生参数 `runtime-env-agent-port=52366`，避免冷启动时 Ray Client 读到端口 0 后等待节点信息超时；不要改成已由 Dashboard Agent HTTP 使用的 52365。
- `TRIBUTO_REDIS_DRIVER_INPUT_B64 is required` 出现在手动运行 Driver 时，表示没有 Consumer 为本次 Job 注入任务输入；应经 Consumer 提交。
- v2.5 对较大输入使用 Ray 原生 `py_modules` 文件传输，仍保留可配置的 Base64 长度限额 262144 字节，避免 Linux 单环境变量限制。Consumer 与 Ray 的包需要一起升级。
- 输出表的 `inferred_at` 应与当前实现一致：无时区 `DateTime` / `DateTime64(6)`。显式时区的类型会被当前 ClickHouse writer 拒绝；本地验收已覆盖该差异。
- 推理查询支持受限的单表标量过滤和 `LIMIT`；当前运行时不支持排序，包含 `ORDER BY` 会明确失败。使用 UI 提交时保持确定性排序关闭；不带排序的 `LIMIT` 不保证每次选中相同实体。
- Sentinel 除自身认证外，还需 Redis 主节点的密码。Sentinel 返回的主节点地址必须从 Head、Worker、Consumer 都可达。
- S3 需要各节点可访问 endpoint 且有 Get/Put/Delete 权限。预检对象使用随机 `tributo-preflight/` 前缀，检查后删除。

失败现场保存在 `--output` 指定目录，包括 Pod 列表、Kubernetes events 和预检日志。

## 5. 重复本地验收

在源码仓库根目录或解压后的部署包根目录执行：

```bash
ACCEPT_IMAGE=knova-tributo-devbox:v2.5 bash scripts/acceptance_local.sh
```

仅启动本机 Docker 内的 Redis、ClickHouse、S3、Ray Head/Worker，不映射宿主机端口，不连接生产服务。
覆盖 30 万行 × 116 特征训练、0/0.5/1 阈值推理、三分类、S3 模型上传、请求错误、凭据/网络错误、超限输入、大输入传输和取消。
结果和 Ray 日志默认写到 `dist/acceptance-时间/`，退出时清理自己的 Compose 资源；设置 `ACCEPT_KEEP=1` 可保留排障环境。
本机需已缓存 compose 中列出的依赖镜像，建议为 Docker 分配至少 16Gi 内存。
训练仅运行少量 boosting rounds 验证链路和数据规模，不代表模型效果或生产吞吐基准。

单独复测 Sentinel 鉴权、节点故障和主节点切换：

```bash
bash scripts/acceptance_sentinel.sh
```

该脚本自动创建并清理自己的 Redis 主从和 3 个 Sentinel，使用一次性的本地测试凭据。
