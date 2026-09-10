# Selected evidence

| Path | Contents |
|---|---|
| `silicon/pico_cycles.json` | Original 120-row RP2040 timing record |
| `demo/records/*.json` | 60 device captures, including 20 prefix continuations |
| `demo/publication.json` | Original and public capture hashes; sanitation rule |

The timing record remains byte-identical to the measurement file:
SHA-256 `8c6966425df28ac196ae0961f59e2b74f992c32466d88f0df436ccb5d71a47b3`.
It measures SRAM execution, not model inference or energy.

Public demo records change only `checkpoint.path` to a repository-relative
location. Programs, traces, geometry, comparisons and measurement fields are
unchanged. A saved record documents a past capture; rebuilding its gallery is
not another hardware experiment.

Generated HTML, caches, datasets, checkpoints and run logs stay ignored.
Protocol and evidence JSON are included selectively, not excluded by extension.
