# Demo media

`hero.gif` records the live page; `rig.jpg` shows the hardware setup. The four
PNG sheets are generated from captured records by `python scripts/demo.py figures`.

- `gallery.png`: category-conditioned device traces.
- `novelty.png`: generated drawings beside sampled training neighbours.
- `prefix.png`: continuations of held-out drawing prefixes.
- `program.png`: bytecode alongside executed geometry.

Black geometry comes from device records. Grey references are host-rendered
training drawings; red marks supplied prefixes. Neighbour and continuation
examples are diagnostic illustrations, not proof of non-memorisation or broad
generalisation. See [demo provenance](../demo.md).

Figures use available system fonts; exact rendered bytes can vary across
platforms. Dataset-derived material retains upstream attribution and licence
terms in [third-party notices](../../THIRD_PARTY_NOTICES.md).
