ARG UV_BUILDER_IMAGE=artifact.nc.rdcloud.4c.hq.cmcc:80/knovapriv-libs/astral/uv:python3.13-trixie
ARG PYTHON_RUNTIME_IMAGE=artifact.nc.rdcloud.4c.hq.cmcc:80/knovapriv-libs/library/python:3.13.9-slim-trixie

# The internal uv image already contains Python, uv, Git, and the CA bundle.
# Package-index settings are supplied by the build environment; credentials are
# intentionally not persisted as Dockerfile ENV values or image layers.
FROM ${UV_BUILDER_IMAGE} AS builder

WORKDIR /app
ENV UV_PYTHON_DOWNLOADS=0 \
    UV_PROJECT_ENVIRONMENT=/opt/tributo-knova

# Keep dependency installation cacheable when application sources change.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# The runtime image is mirrored inside RDCloud and already contains the Debian
# trixie C/C++ runtime libraries required by the prebuilt Python wheels.
FROM ${PYTHON_RUNTIME_IMAGE}

ARG VERSION=0.1.0
LABEL org.opencontainers.image.title="tributo-knova" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.source="https://github.com/Chang-Tong/tributo-knova"

COPY --from=builder /opt/tributo-knova /opt/tributo-knova
ENV PATH="/opt/tributo-knova/bin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN set -eu; \
    site_packages="$(/opt/tributo-knova/bin/python -c \
        'import sysconfig; print(sysconfig.get_paths()["purelib"])')"; \
    mkdir -p \
        /opt/tributo-runtime/core \
        /opt/tributo-runtime/extensions \
        /etc/tributo-knova; \
    cp -a "${site_packages}/tributo" \
        /opt/tributo-runtime/core/tributo; \
    cp -a "${site_packages}/tributo_knova" \
        /opt/tributo-runtime/extensions/tributo_knova; \
    cp -a "${site_packages}/tributo_broker_redis" \
        /opt/tributo-runtime/extensions/tributo_broker_redis; \
    cp -a "${site_packages}/tributo_algorithms_boosting" \
        /opt/tributo-runtime/extensions/tributo_algorithms_boosting; \
    useradd --uid 10001 --user-group --create-home \
        --home-dir /var/lib/tributo-knova --shell /usr/sbin/nologin \
        tributo-knova

COPY deploy/config/knova-compose.json /etc/tributo-knova/config.json

USER tributo-knova
HEALTHCHECK --interval=30s --timeout=8s --start-period=15s --retries=3 \
    CMD ["tributo-knova", "health", "--config", "/etc/tributo-knova/config.json"]
CMD ["tributo-knova", "consume", "--config", "/etc/tributo-knova/config.json"]
