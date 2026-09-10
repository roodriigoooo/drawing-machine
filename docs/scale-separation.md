# Scale separation — the test that validates or kills the hierarchy

Moved out of `PLAN.md` on 2026-08-05. `PLAN.md` §9.6 keeps the one-paragraph
version and the decision it makes; this is the instrument design.

The claim under test (`docs/diffusion-strategy.md` §5.1–5.3) is that denoising
decomposes by scale and autoregression cannot, and that local structure transfers
up while global structure does not. `dm/data/synthetic.py:LADDER` already varies
nesting depth independently of stroke count, so the test needs no new data:

> **Train on `simple`, evaluate on `elaborate`, and measure local and global
> fidelity separately. Prediction: local gap small, global gap large. If both
> degrade equally, the levels are not separable and the hierarchical design is
> wrong.**

## Design

- The AR model reconstructs nothing, so the paired traces come from
  **prefix-conditioned completion**: prompt with the first k% of a held-out
  program's symbols, let the model finish, compare the resulting geometry
  against the true full program. `generate()` needs a `prompt` argument —
  `Attention.forward` already handles the prefill case, and the halt monitor
  must be fed the prompt's bytes so its parse state starts correct. Prompts
  must be a whole number of bytecode bytes or the bit arm's monitor
  desynchronises.
- **Global** = Chamfer on stroke centroids. Ignores stroke shape entirely.
- **Local** = optimal assignment of strokes between the two traces on
  centroid distance, then each matched pair centred on its own centroid,
  arclength-resampled, and compared pointwise (min over forward/reversed, so
  draw direction is not counted as shape error).
- Report unmatched strokes as a separate count ratio. Folding them into
  either number would contaminate the very separation being measured.
- Matching needs `scipy.optimize.linear_sum_assignment`; add `scipy` to
  `pyproject.toml` when this lands. Greedy matching makes the metric depend
  on iteration order and would bias the decision this test exists to make.
- Train on `simple`, evaluate on `elaborate`. A large global gap with a small
  local gap validates the hierarchical design; equal degradation falsifies
  it. **Do this before building the planner**, not after.

## Added 2026-08-05, when the metrics landed

Three decisions the original design did not state, all of which change what the
result means.

- **Each number needs a chance-level reference on the same rung.** `elaborate`
  drawings have three times the strokes of `simple` ones, so the *scale* of both
  Chamfer and the matched-shape distance changes with the rung even for a
  perfect model. "Global got worse" would then be partly a statement about the
  metric. Report `1 - d(model, truth) / d(chance, truth)`, where chance is the
  same metric between the truth and an unrelated program from the same rung.
  Skill scores are comparable across rungs; raw distances are not.
- **The centroid is the exact arclength-weighted centre**, not the mean of the
  vertices. The VM emits 16 points for a flattened curve and 2 for a line, and a
  closed polygon repeats its first vertex, so a vertex mean moves the *layout*
  number for reasons that are purely about how the path was written down. The
  segment-length-weighted mean of segment midpoints is exact, cheaper than
  resampling, and moves by exactly zero when a vertex is added mid-segment.
- **`local` centres on that same centroid.** Otherwise the offset the layout
  number measures leaks into the shape number, and the two stop being
  independent — which is the only property the experiment actually needs.

`dm/eval/scale.py` implements this and `tests/test_scale.py` pins the
independence directly: translating the whole drawing must move `global` and not
`local`, reshaping in place must move `local` and not `global`, and draw
direction and vertex density must move neither.

## Status

- **Done**: `generate(prompt=...)` (`dm/models/transformer.py`), the metrics
  (`dm/eval/scale.py`), `scipy` in `pyproject.toml` as the `eval` extra.
- **Next**: the driver — train on `simple`, complete held-out `elaborate`
  prefixes, and report both skill scores per rung with unmatched counts beside
  them.

---

## The result, coverage-corrected

_Moved out of `PLAN.md` when that file was compacted on 2026-08-08. Section
references of the form §N are to `PLAN.md`._

| rung | strokes truth/pred | coverage | global skill | local skill |
|---|---|---|---|---|
| simple | 5.2/5.2 | 0.55 | +0.285 | +0.383 |
| busy | 18.2/7.3 | 0.44 | +0.177 | +0.474 |
| nested | 8.6/5.5 | 0.52 | −0.105 | +0.064 |
| elaborate | 43.0/12.1 | 0.31 | **+0.004** | **+0.290** |

**Local exceeds global at every rung, and global collapses to chance on the two
hardest while local survives** — the premise a hierarchical planner needs. Two
caveats: the model saw only `simple`, so the hard rungs are also
out-of-distribution, and it emits 12 strokes where the truth has 43, which is
§9.5a at stroke level.

