# v2.4 日志复查

日期：2026-09-24。全部使用本机独立 Redis、Ray Head、Ray Worker 和已交付的 ARM64 v2.4 镜像，没有连接生产服务或修改镜像。

## 结果

| 检查 | 实测结果 |
|---|---|
| 请求 schema 错误 | `ray job logs` 有 Driver 启动、输入关联信息和完整 Traceback；Redis 收到一条 EXECUTION_FAILED |
| Ray 缺少 Redis 凭据 | CLI 有 RedisCredentialUnavailable 堆栈；Consumer 补报一条 RAY_JOB_FAILED |
| 任务失败后 Consumer 重启 | 恢复原 submission 并补报；再次重启不重复发送终态 |
| 手动启动 Driver 而没有输入 | stderr 明确打印缺少输入的堆栈，退出码 1 |
| 持续运行的控制 Job | 从 PENDING 到 RUNNING 再故意失败；CLI 可读取完整 RuntimeError 堆栈 |
| Consumer / Ray 源码一致性 | Head、Worker 各自的安装目录和分发目录逐文件一致 |
| 相关单元测试 | 25 项通过 |
| 交付镜像包 | SHA256 与原校验文件一致 |

三项业务故障回归均直接执行了 `ray job logs`，不是仅查看 Dashboard 或测试进程输出。测试密码值未出现在这些 CLI 日志和 Consumer 日志中。

## 必须澄清：两种 Ray ID

`submission_id` 在提交时就确定，Consumer 与 Driver 都打印它，CLI 的 logs/status 可以直接使用它。

`ray_job_id` 是 Ray Driver 内部 ID，可能稍后才出现。控制 Job 的真实观测顺序为：

```text
PENDING  ray_job_id=null
RUNNING  ray_job_id=null
RUNNING  ray_job_id=02000000
FAILED   ray_job_id=02000000
```

本轮快速失败任务的 Consumer 日志为 `ray_job_id=<unknown>`。现有 Consumer 在提交及恢复时采集这个字段，没有持续刷新每条已经打印的日志。因此不能承诺每个任务的 Consumer 日志都有内部 Ray ID。

复查脚本最初额外断言每个失败 Job 都必须存在内部 ID，这条断言实际失败。已保留原检查脚本及现场；后续单独核验确认日志关联、完整堆栈及失败回传通过，内部 ID 缺失也明确记录，没有把它替换为伪造值。

排障时执行：

```bash
ray job logs --address http://127.0.0.1:8265 <submission_id>
ray job status --address http://127.0.0.1:8265 <submission_id>
```

Driver 尚未启动或被直接终止时，仍可能没有应用堆栈，应结合 Job status 和 Consumer 补报。该边界不会因为增加应用日志而消失。

本轮测试资源已清理。完整现场在源码仓库 `dist/logging-recheck-20260924/`；本部署包的 `evidence/logging-recheck/` 包含关键日志及结果。
