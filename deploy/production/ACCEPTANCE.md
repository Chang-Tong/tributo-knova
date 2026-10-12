# v2.4 本地验收报告

日期：2026-09-24。测试完全在本机 Docker Desktop ARM64 和临时 Kind 集群执行。

## 交付物

- 镜像：`knova-tributo-devbox:v2.4`，`linux/arm64`，Ray 2.55.1 / Python 3.13。
- 镜像 ID：`sha256:50100dc136124bfe22ced8a8790aa58075deb2bb6dcd7bd007d14426ee502d95`。
- zstd 文件：`knova-tributo-devbox-arm64-v2.4.tar.zst`，536,303,763 字节。
- SHA-256：`02795a00ce06b250522273d4a4bc1d732cba9f336435e21e3fee6076896673bd`。
- 已校验 zstd 完整性、归档 manifest、唯一镜像标签和 ARM64 架构。
- 部署入口：填写 settings.json 后执行 `python3 deploy.py apply --settings settings.json`。

## 真实服务测试：17 / 17 通过

使用实际 Redis Streams → 打包后的 Consumer → Ray Head/Worker → ClickHouse / S3，未替换训练算法为 mock。
Ray Head 与 Worker 位于不同容器，Consumer 为独立进程。任务输入和结果事件均来自真实 Redis。
训练数据是合成的 300,000 行、116 个 Float32 特征，另有标签及标识列；实际训练/验证/测试分别为 210,000 / 45,000 / 45,000 行。
每次训练使用少量 boosting rounds 验证链路，不是生产吞吐或模型效果基准。下面时间包含提交、准备环境和事件等待。

| 场景及预期 | 结果 | 秒 |
|---|---|---:|
| 30 万行、116 特征分布式训练 | 通过 | 138.72 |
| 阈值 0 推理：300 行、300 个正类 | 通过 | 54.19 |
| 阈值 0.5 推理：300 行、150 个正类 | 通过 | 47.12 |
| 阈值 1 推理：300 行、0 个正类 | 通过 | 71.89 |
| 阈值升高时正类数量单调不增 | 通过 | 0.0 |
| 116 特征三分类训练 | 通过 | 101.02 |
| 分布式训练及 S3 模型发布 | 通过 | 92.24 |
| 未知算法：FAILED 且有堆栈 | 通过 | 17.0 |
| 不存在的特征：FAILED 且有堆栈 | 通过 | 19.15 |
| 空特征列表：FAILED 且有堆栈 | 通过 | 13.86 |
| ClickHouse 密码错误：FAILED 且有堆栈 | 通过 | 17.91 |
| ClickHouse 连接失败：FAILED 且有堆栈 | 通过 | 17.86 |
| 存储路径不可写：FAILED 且有堆栈 | 通过 | 42.51 |
| 超出 256 KiB：Consumer 明确拒绝 | 通过 | 0.56 |
| 约 200 KiB：成功加载输入，再按预期校验失败 | 通过 | 14.53 |
| 提交前取消：CANCELLED | 通过 | 0.51 |
| 提交后取消：CANCELLED | 通过 | 9.49 |

大输入用例包含 150,000 字节附加元数据，整体 Base64 大约 200 KiB。测试断言 Driver 实际加载到输入、有预期的业务校验堆栈，而不是把 Ray 启动失败误算为通过。
超限用例约 282,000 字节，明确返回 DRIVER_INPUT_TOO_LARGE。116 个简单特征描述的输入仅约 7 KiB；列数本身不能推出 payload 大小。

## 其他实际故障演练

- Sentinel 8 / 8：正确认证、Redis DB 3、单 Sentinel 不可达、Sentinel 密码错误、Redis 密码错误、两种 Secret 缺失、错误 master 名、Redis 主节点停止后自动恢复。
- Sentinel 已用独立脚本完整重跑；初次使用会消失的 Docker DNS 名称及单文件挂载的搭建问题，改为稳定节点地址、可原子重写的目录挂载后通过。
- 日志回归 3 / 3：Driver 失败有堆栈；Driver 无法连接 Redis 时 Consumer 补报；任务已失败后 Consumer 重启仍补报，并验证重复恢复不产生多条终态。
- 镜像一致性：正常通过；故意改旧分发目录代码、故意删分发模块，均被检查拒绝。
- Head 仅 3Gi 时真实触发 Ray 的内存保护，JobSupervisor 被杀、Driver 日志为空；Consumer 仍补报 FAILED。恢复足够内存并重启 Head 后通过。部署默认每个 Ray Pod 8Gi，本地分离容器测试为 6Gi。
- 带 UTC 时区的 ClickHouse inferred_at 与当前无时区输出不兼容，真实复现写入错误。验收表改为生产实现对应的无时区类型后，三档阈值均逐行通过。

## Kubernetes 实际部署

Kind Kubernetes 1.35.8 + KubeRay 1.6.0，使用已校验的 v2.4 镜像：

- 服务端 dry-run、namespace/架构/Secret 检查通过；缺少 Secret 键、无对应架构节点时在变更前拒绝。
- 普通预检 Pod 直接 ray.init 的错误在本地暴露，已改为 Ray Jobs 原生提交；预检实际在 Head 和 Worker 上检查 Redis 和源码一致性。
- 新 Consumer 在预检完成前保持 0 副本；一次真实预检失败时，旧 Consumer 保持运行。
- 部署通过后：新 Consumer 1/1，旧 Consumer 0/0。
- 重复 apply 后 Consumer Pod UID 不变，没有重新部署。
- Secret 键存在但值错误时，预检 Job 按预期失败。
- 实际 rollback 后：新 Consumer 0/0，旧 Consumer 恢复 1/1。

## 单元测试及明确限制

- KnoVa：106 项通过，其中部署工具 17 项；大输入传输辅助测试 11 项通过。
- 另选的 Broker 传输/生命周期集合为 22 项通过、3 项失败。失败来自通用训练适配器仍引用已不存在的 `tributo.training.xgboost_trainer`，属于旧 generic API 与当前 Core 的不兼容；KnoVa 使用自己的训练适配器，以上真实链路不走该入口。若部署的是 `tributo-broker-redis` 通用 CLI，不能套用本报告的通过结论。
- 本轮没有启动 ai-model-control/webui 做浏览器验收；验证了 Consumer、Ray 和事件协议端到端。
- Kubernetes 的 S3/PVC 预检选项已提供，真实 S3 模型上传和共享目录在 Docker 环境跑过；未连接生产 S3/NFS，也未验证生产 CSI、网络策略、DNS、资源配额或特定数据。
- 这些复现说明多种同类失败可被诊断，不能据此断定此前生产故障一定是内存不足。

## 复测与证据

- `scripts/acceptance_local.sh`：本地真实训推验收。
- `scripts/acceptance_sentinel.sh`：本地 Sentinel 故障切换验收。
- `scripts/reproduce_driver_failure.py`：日志与终态恢复回归。
- 原始完整证据在源码仓库 `dist/production-acceptance/`，包含各任务 Ray 日志、Redis 事件、Ray status、逐项结果和 Kubernetes 部署/回滚快照。
- 交付包 evidence 目录包含汇总结果、关键日志及测试计数。旧失败轮次保留在完整证据目录，没有被覆盖成通过。
