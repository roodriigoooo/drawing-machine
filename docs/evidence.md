# Evidence boundaries

The [README](../README.md) and [Hypotheses and Results](hypothesis_and_results.md) summarize results. This page states what those results do not establish.

- **Representation:** Paired intervals over drawings are not uncertainty across model seeds. Synthetic and QuickDraw comparisons have different seed counts.
- **Structure and context:** Teacher-forced likelihood, controlled preference, and free-running exact success are separate measurements. Strong conditional preference does not imply reliable completed drawings.
- **Feedback:** The tested implementation failed generic and stability gates before relation evaluation. That rules out this tested package, not all feedback mechanisms.
- **Hardware:** The RP2040 runs the virtual machine, not the neural network. The documented conformance sweep matched 12,670 traces; the preserved timing record contains 120 programs. Footprint excludes the host, transport harness, and drawing storage. Energy consumption remains unmeasured.
- **Demo:** 60 saved captures support device execution and reference equality. Selected images do not prove generalization, absence of memorization, or set-level quality. Nearest-neighbor comparisons use a sampled reference bank.

Methods and detailed numbers are documented in [Hypotheses and Results](hypothesis_and_results.md); [demo records](demo.md) have separate provenance.

The experimental `copy-relation` branch has engineering tests but no learned outcome claimed. Engineering readiness is not scientific qualification.
