# References

These papers shaped the representation, model designs and experiments. Borrowing
an idea is not a replication of its reported results; the notes below distinguish
the two. Paper identifiers refer to the original publications, not bundled PDFs.

Ideas taken:

- **sketch-rnn** (1704.03477, Ha & Eck): stroke-native representation and the
  implicit pen state that `MOVE`/`LINE` mirror. Supplies the Tier B baseline.
- **DreamCoder** (3453483.3454080, Ellis et al.): wake-sleep library learning
  is the model for the L2 `CALL` tier.
- **DeepSVG** (2007.11301, Carlier et al.): hierarchical vector generation;
  source of the Tier C dataset.
- **diffvg** (Li et al.): differentiable rasterisation, for the diffvg loss
  term in the metric suite.
- **Learning to Infer Graphics Programs from Hand-Drawn Images**
  (1707.09627, Ellis et al.): the program-vs-trace distinction that motivates
  separating L0 from L1.

Ideas challenged:

- **IconShop** (2304.14400): fused `(op, x, y)` tokenisation. At `d=256` a
  4096-entry table is 1.05M parameters: the entire budget before a single
  layer. The claim under test is that this default is simply unaffordable at
  sub-1M, and that byte and bit alphabets, which are free, may be preferable.
- **StrokeNUWA** (2401.17093): learned stroke tokens. Same objection plus a
  second: a learned tokenizer is a second model whose parameters are rarely
  counted against the budget.

Ideas evaluated or replicated:

- **Full-bandwidth transformer** (2608.08888, Wang et al.): source for
  gated latent feedback, multi-pass objective, prefix mixin and
  stability schedule. The project implemented the feedback seam but did not
  replicate the source condition: 857.6k parameters/83.1M content symbols under
  a project optimizer and batching regime versus the paper's 1B/200B stability
  run. The implementation closed at project scale before relation evaluation; see
  [Hypotheses and Results](hypothesis_and_results.md#gated-latent-feedback).
- **SPIRAL** (1804.01118) and **Learning to Paint** (1903.04411): RL program
  synthesis for images. Not replicated; they are the comparison point for why
  this project uses likelihood on ground-truth programs (Tier A) instead of RL.
- **Rendering-Aware RL for Vector Graphics** (2505.20793): recent
  rendering-aware objective relevant to stroke planning.

> Honest status: these are the papers on disk and the roles they are intended to
> play. Sketch-rnn's representation, DreamCoder's library-learning idea and the
> full-bandwidth transformer's feedback seam have shaped code. None of their
> headline numerical results has been replicated, and nothing here should be
> replicated for its own sake.
