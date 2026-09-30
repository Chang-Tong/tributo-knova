# v2.5 完整镜像与最小增量交付

以下命令用于复现镜像归档或增量构建，从源码仓库根目录执行。正式交付归档的实际大小与 checksum 以更新后的 `release.json` 为准。

最终交付镜像：`knova-tributo-devbox:v2.5`，`linux/arm64`，ID 为 `sha256:9f9c8077ebe4f5971380d43b363b7c1494f04a3f39861ec0a0cf2e8974569101`。完整 zstd 归档为 536462571 字节（约 512 MiB），内部 RepoTags、Config 和平台已核对。本机依赖预检见 [PREFLIGHT-V25.md](PREFLIGHT-V25.md)；生产 Kubernetes 调度、PVC、网络和切换仍需现场验证。

## 方案 A：交付完整镜像

适用于目标只需导入已验证镜像的场景。先核对标签指向的精确镜像 ID，现有打包脚本执行 `docker save`、zstd 19 压缩、完整性校验，并写 SHA-256 和平台文件：

```bash
docker image inspect knova-tributo-devbox:v2.5 --format '{{.Id}} {{.Os}}/{{.Architecture}}'
test "$(docker image inspect knova-tributo-devbox:v2.5 --format '{{.Id}}')" = \
  'sha256:9f9c8077ebe4f5971380d43b363b7c1494f04a3f39861ec0a0cf2e8974569101'
bash scripts/package_image.sh knova-tributo-devbox:v2.5 dist/knova-tributo-devbox-arm64-v2.5.tar.zst
```

交付这三个文件，以及 `deploy/production/` 部署工具目录：

```text
knova-tributo-devbox-arm64-v2.5.tar.zst
knova-tributo-devbox-arm64-v2.5.tar.zst.sha256
knova-tributo-devbox-arm64-v2.5.tar.zst.platform
```

导出后再把实际归档大小和 checksum 写入 `release.json`。历史 `ACCEPTANCE.md` 保留 v2.4 标识；真实服务器配置、Secret 文件及本机 `dist/.../runtime.env` 单独保管。

在**每个可能调度工作负载的 Kubernetes 节点**上导入：

```bash
bash import-image.sh knova-tributo-devbox-arm64-v2.5.tar.zst containerd
```

Docker 节点把最后参数改成 `docker`；nerdctl 节点改成 `nerdctl`。脚本校验 checksum 和 zstd 后导入相应运行时。settings 的 `image` 使用归档中的 `knova-tributo-devbox:v2.5`，新发布名和输出目录例如 `tributo-knova-v25-final` / `release-v25-final`。正式发布后保持该版本标签指向同一个镜像。

## 方案 B：目标已有 v2.4 时离线增量重建

已生成 `knova-v2.5-hotfix-context.tar.gz`，109636 字节（约 107 KiB），SHA256 为 `6e882d60feb5e5c2b964850fd504d8787887c03ad0ec03e560f864990ae98741`。它是离线构建上下文，需要目标已有下面指定的基础镜像。本地已用该归档作为唯一构建上下文、`--network=none` 构建成功，并核对 KnoVa、broker、boosting、Core、ray-clickhouse 共 349 个 Python 文件与交付镜像一致；该重建的镜像 ID 为 `sha256:5f3b902ab8cf11beb4d8107d3c98020e3ce4b0087d662a985660191e02fa6783`。完整业务验收针对交付的 `9f9c8077…` 镜像，目标自行重建后仍应执行同样验收。

此方案传输源码与两个 wheel，目标使用本地基础镜像构建新镜像。支持的基础镜像是本轮使用的 ARM64 / Python 3.13 / Ray 2.55.1 v2.4：

```text
sha256:50100dc136124bfe22ced8a8790aa58075deb2bb6dcd7bd007d14426ee502d95
```

最小兼容更新包括 KnoVa、Redis Broker、Boosting。`Dockerfile.hotfix` 同步更新镜像安装目录和 Ray 分发目录，并执行三包逐文件校验。Head、Worker、Consumer 使用同一个新镜像。本兼容范围是 KnoVa 自定义适配器，默认 `tributo-generic-v1` 训练入口的已知 Core 模块缺失未修复。

在源码机准备白名单构建上下文：

```bash
mkdir -p dist
tar --exclude='__pycache__' --exclude='*.pyc' -czf dist/knova-v2.5-hotfix-context.tar.gz \
  Dockerfile.hotfix src/tributo_knova deploy/verify_image.py \
  vendor/wheels/tributo_broker_redis-0.1.0-py3-none-any.whl \
  vendor/wheels/tributo_algorithms_boosting-0.1.0-py3-none-any.whl \
  vendor/wheels/tributo-algorithms-LICENSE
(cd dist && shasum -a 256 knova-v2.5-hotfix-context.tar.gz > knova-v2.5-hotfix-context.tar.gz.sha256)
```

在目标构建机核对 checksum、精确基础镜像 ID，再离线构建：

```bash
sha256sum -c knova-v2.5-hotfix-context.tar.gz.sha256
test "$(docker image inspect knova-tributo-devbox:v2.4 --format '{{.Id}}')" = \
  'sha256:50100dc136124bfe22ced8a8790aa58075deb2bb6dcd7bd007d14426ee502d95'
mkdir -p knova-v2.5-hotfix-context
tar -xzf knova-v2.5-hotfix-context.tar.gz -C knova-v2.5-hotfix-context
cd knova-v2.5-hotfix-context
docker build --pull=false --network=none --platform linux/arm64 \
  --build-arg BASE_IMAGE=knova-tributo-devbox:v2.4 \
  --build-arg RELEASE_VERSION=2.5 \
  -f Dockerfile.hotfix -t knova-tributo-devbox:v2.5-local .
docker image inspect knova-tributo-devbox:v2.5-local --format '{{.Id}} {{.Os}}/{{.Architecture}}'
```

该构建仅安装随包 wheel，使用 `--no-index --no-deps` 继承基础镜像依赖。重建产物有独立镜像 ID，记录后执行相同预检和真实训练 / 推理验收，再分发到全部节点。已有内部镜像仓库时，推送 / 拉取复用相同基础层，Docker 会传输缺少的层。

## 发布与回滚

两种方案均使用 [README.md](README.md) 的同一部署入口。先暂停上游并等待旧 Ray Job 完成，再通过新发布名 / 输出目录执行 `apply`；填写 `previous_consumer` 后保留 `switch.json` 和 `manifest.json` 供回滚。

任务 JSON、Redis stream/group、存储配置沿用现场配置，凭据通过 Head、Worker、Consumer 共用的 Kubernetes Secret 注入。训练运行时读取 `extensions.tributo.training_runtime.ray` 的 `num_workers`、`cpus_per_worker`、`use_gpu`；当前 UI 的资源控件未映射到这些字段，不能根据 UI 选择断言资源已生效。集群 Pod 的规模由部署 settings 的 workers / cpus / memory 确定。
