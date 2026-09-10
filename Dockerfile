FROM python:3.12-slim AS builder

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir uv==0.9.26

WORKDIR /app
ENV UV_PROJECT_ENVIRONMENT=/opt/tributo-knova
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

FROM python:3.12-slim

ARG VERSION=0.1.0
LABEL org.opencontainers.image.title="tributo-knova" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.source="https://github.com/Chang-Tong/tributo-knova"

COPY --from=builder /opt/tributo-knova /opt/tributo-knova
ENV PATH="/opt/tributo-knova/bin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install --yes --no-install-recommends adduser \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p \
        /opt/tributo-runtime/core \
        /opt/tributo-runtime/extensions \
        /etc/tributo-knova \
    && cp -a /opt/tributo-knova/lib/python3.12/site-packages/tributo \
        /opt/tributo-runtime/core/tributo \
    && cp -a /opt/tributo-knova/lib/python3.12/site-packages/tributo_knova \
        /opt/tributo-runtime/extensions/tributo_knova \
    && cp -a /opt/tributo-knova/lib/python3.12/site-packages/tributo_broker_redis \
        /opt/tributo-runtime/extensions/tributo_broker_redis \
    && cp -a \
        /opt/tributo-knova/lib/python3.12/site-packages/tributo_algorithms_boosting \
        /opt/tributo-runtime/extensions/tributo_algorithms_boosting \
    && addgroup --system --gid 10001 tributo-knova \
    && adduser --system --uid 10001 --ingroup tributo-knova \
        --home /var/lib/tributo-knova tributo-knova

COPY deploy/config/knova-compose.json /etc/tributo-knova/config.json

USER tributo-knova
HEALTHCHECK --interval=30s --timeout=8s --start-period=15s --retries=3 \
    CMD ["tributo-knova", "health", "--config", "/etc/tributo-knova/config.json"]
CMD ["tributo-knova", "consume", "--config", "/etc/tributo-knova/config.json"]
