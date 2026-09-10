# Evidence boundaries

The [README](../README.md) summarizes results. This page states
what those results do not establish.

- **Representation:** paired intervals over drawings are not uncertainty across
  model seeds. Synthetic and QuickDraw comparisons have different seed counts.
- **Structure and context:** teacher-forced likelihood, controlled preference
  and free-running exact success are separate measurements. Strong conditional
  preference does not imply reliable completed drawings.
- **Feedback:** the tested implementation failed generic/stability gates before
  relation evaluation. That rules out this tested package, not all feedback.
- **Hardware:** the RP2040 runs the VM, not the neural network. The documented
  conformance sweep matched 12,670 traces; the preserved timing record contains
  120 programs. Those are different sample sets. Footprint excludes the host,
  transport harness and drawing storage. Energy remains unmeasured.
- **Demo:** 60 saved captures support device execution and reference equality.
  Selected images do not prove generalisation, absence of memorisation or
  set-level quality. Nearest-neighbour comparisons use a sampled reference bank.

Methods live with the studies; [hardware measurements](claim4-bringup.md) and
[demo records](demo.md) have separate provenance.

The experimental `copy-relation` branch has engineering tests but no learned
outcome claimed. Engineering readiness is not scientific qualification.
