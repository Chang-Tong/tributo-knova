# tributo-knova

KnoVa integration package built on the public `tributo` and
`tributo-broker-redis` packages.

This repository starts as a thin integration layer. It should reuse the two
upstream packages directly and add only KnoVa protocol mapping, missing runtime
behavior, and deployment assets.

## Development

The local development configuration resolves both dependencies from sibling
repositories:

```bash
uv sync --extra dev
uv run pytest
```

