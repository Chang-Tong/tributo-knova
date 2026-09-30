# v2.7 Consumer-only 本地验收记录

日期：2026-09-25。全部在本机 Docker 与隔离 Kind 执行，未连接生产集群。
v2.6 的常量特征二分类 ONNX base_score 导出缺陷已使旧镜像退出最终交付；本包引用新镜像 v2.7。
本包仅涵盖部署/自查及对应小业务验收，完整 116 特征业务矩阵另见主验收报告。

## v2.7 实际完成

- 镜像 `knova-tributo-devbox:v2.7`，linux/arm64，config ID 为 `sha256:b49ce75601f0c845063ad4bb9e0d4190df488a53b8145dfc6cdf24249f7388cb`。
- Consumer 复用既有 Redis Sentinel、ClickHouse、S3、Ray Head/Worker。Ray 使用不同的旧镜像，且没有 `/opt/tributo-runtime`；新源码由 py_modules 分发并由 doctor 检查。
- 5 项真实 CLI 核心场景全部通过：check、apply、相同配置幂等、rollback、再次 apply。初次预检约 61 秒，apply 约 73 秒；现场耗时取决于镜像、网络和节点。
- 幂等 apply 保持 Pod UID 不变。回滚后旧 Consumer Ready=1、新 Deployment 删除；再次 apply 后旧副本 0、新 Consumer Ready=1。
- 严格非空比较 Redis master/replica/sentinel、ClickHouse、S3、Ray head/worker 共 7 个 Deployment 的 UID+spec，整个周期完全相同。自查会产生并清理随机临时 key/object，此项仅声明 Kubernetes Deployment 清单未变。
- 真实独立 Consumer CLI：2 个常量数值特征，标签 class 1 占 20%，正类选 class 0。训练 1,000 行耗时 180.36 秒，推理 500 行耗时 72.20 秒，两个 Ray Job 均 SUCCEEDED，Redis 各有唯一 COMPLETED 终态。
- 原生 oracle 按 entity_id 重新读取 ClickHouse 源 f1/f2，对全部 500 个输出行验证。UBJ 原生 p1=0.200000003；ONNX 正类 0 概率与 1-p1 最大误差 1.49e-08。
- UBJ 与独立 feature_names.json 的长度、SHA256、特征顺序均按本次训练 manifest 校验；SHAP 每项贡献、base value 和原生选定 margin 最大误差 0，sigmoid 重建概率最大误差 1.13e-08，源特征回显误差 0。
- Consumer operation→Ray submission 映射、Driver 实际模块路径、输入、runtime 和完成日志齐全。推理成功 INFO 的 status 不再为 unknown。
- 最终再次核验实际 Pod imageID 等于冻结候选的 CRI config ID。本轮 Kind 已删除并确认节点容器不存在。

## 记录的验收脚本修正

不隐去测试工具本身的错误：

1. 首次镜像核验错误地要求 Pod imageID 一定是 repoDigest。本轮导入没有 repoDigest，实际 imageID 就是正确的 CRI config ID；修正为核定 config ID 后接受它或同一 CRI 的 repoDigest。
2. 首次后置 oracle 错误地要求 UBJ 内嵌 feature_names。实际产物按契约由独立 feature_names.json 承载。修正后校验该文件及 UBJ 的 SHA/顺序，再对原来已成功的 500 行重放 oracle，没有重跑或额外计数训练/推理。

以上是本地验收 helper 修正，生产部署脚本及业务代码在本子任务中未改动。安全详情见 evidence/harness-corrections.json。

## 历史部署脚本回归

`deploy.py`、`doctor.py`、`import-image.sh` 的逻辑及 SHA256 与 v2.6 脚本验收时一致。
历史 66 项部署/doctor 单测、19 项 Kind 故障/回滚场景保留为脚本回归证据，明确标为 v2.6；
本次 v2.7 另外执行上述 5 项核心镜像路径，不能把旧镜像的 19 项写成在 v2.7 全部重跑。
早期 v2.6 空集组件选择器断言已作废；历史最终周期及本次新周期均使用真实 7 组件比较。

## 边界与上线要求

- 本包只含占位符示例和安全 JSON 摘要，不含 kubeconfig、真实运行配置、凭据、恢复 journal、原始日志。
- v2.7 镜像 zstd 包由主流程实际通过 import-image.sh 加载至 Docker。containerd/nerdctl 路径未分别做完整导入验证。
- doctor 临时 Redis 前缀可写不证明业务 task/event/cancel 前缀 ACL；CHECK GRANT INSERT 不证明输出列类型/时区兼容。
- 生产恢复任务投递前，仍需现场真实小训练、带/不带 SHAP 推理及终态/落库验证。没有将本地成功写成生产已验收。
