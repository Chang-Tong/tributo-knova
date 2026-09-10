# tributo-knova

KnoVa integration package built on the public `tributo` and
`tributo-broker-redis` packages.

This repository starts as a thin integration layer. It should reuse the two
upstream packages directly and add only KnoVa protocol mapping, missing runtime
behavior, and deployment assets.

The current bootstrap accepts the existing KnoVa protocol v2 training and
batch-inference envelopes, then adapts them to the public Redis broker runtime.
Redis Streams consumption, consumer groups, pending recovery, cancellation,
Ray Job admission, retries, and acknowledgements remain owned by
`tributo-broker-redis`.

See [the implementation plan](docs/implementation-plan.md) for the four
milestones. The current branch completes the request/admission/event boundary;
the KnoVa training and inference Ray driver is the next vertical slice and is
not release-ready yet.

## Development

The local development configuration resolves both dependencies from sibling
repositories:

```bash
uv sync --extra dev
uv run pytest
```
