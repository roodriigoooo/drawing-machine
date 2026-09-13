# Methodology

This project isolates how representational choices affect small transformers writing drawing programs. To ensure valid comparisons, architectures, alphabets, and evaluation budgets are held to strict experimental controls.

---

## 1. The Four Ablation Axes

Prior drawing tokenizers often confound vocabulary size, coordinate resolution, and grammar constraints. Here, four axes are separated and varied independently:

| Axis | Levels | What it Isolates |
|---|---|---|
| **Typing** | `token` vs `byte` | Exactly one bit of structure per position: is this symbol an opcode? Vocabularies are 269 vs 258, sequence lengths are identical. |
| **Granularity** | `byte` vs `bit` | Field-width and boundary discovery. 8x sequence length, identical information content. |
| **Fusion** | `token` vs `token_typed` | Tests whether pre-binding operand kinds helps or merely inflates the embedding table (274 to 1,554 symbols at ISA v2). |
| **Relativity** | `abs` vs `delta` | Coordinates as absolute canvas positions vs offsets from the pen. Same alphabet, length, and information. |

### Codec Properties
- **Identical information:** Coordinates are 8-bit unsigned integers. The relative view is an exact mod-256 bijection with absolute coordinates. All eight codecs encode identical geometric information with zero quantization confound, verified by roundtrip tests across all corpora.
- **Relativity isolates translation:** Under absolute coordinates, translating a drawing shifts every coordinate byte in the sequence. Under relative delta coordinates, a translation modifies exactly one coordinate pair at the first move.
- **Cross-codec metrics:** Loss per token is not comparable across alphabets (a bit alphabet naturally yields lower cross-entropy because its vocabulary size is 4). All models are evaluated in **bits per drawing** ($NLL \times \text{sequence\_length} / \ln 2$).
- **Uniform architecture:** Models use pre-norm transformer blocks with RMSNorm, RoPE positional embeddings, SwiGLU activations, and tied input-output embeddings. This ensures that observed differences stem from the data representation rather than architectural tricks.

---

## 2. Evaluation Regimes and Asymptotes

Comparing sequence lengths that differ by 8x requires careful compute budgeting. No single budget rule can place two arms at the same point on their loss curve:

- **Token-matched:** Equal total tokens seen during training. Answers what a fixed data budget buys.
- **Step-matched:** Equal optimizer update steps. Answers what a fixed compute budget buys.
- **Converged:** Trained until validation loss slope flattens ($|\text{tail}| < 0.5\text{ bits per 1,000 steps}$ over a fixed fraction of the schedule). Only converged comparisons measure the true representational cost of an alphabet.

Validity is treated as a continuous measurement rather than an exception. Malformed programs produced during early training are recorded as execution faults within the virtual machine, avoiding cherry-picking or post-hoc filtering.

---

## 3. Dataset Tiers

Rather than treating data as a uniform pool, datasets represent distinct structural levels:

- **Tier A (Synthetic):** Procedurally generated curves and polylines with known ground-truth structure. Used to measure loop recovery (`REPEAT`), coordinate precision, and length generalization.
- **Tier B (QuickDraw):** Human doodles from Google Quick, Draw! filtered to five balanced categories: cat, bus, flower, sailboat, and bicycle (375,000 total drawings: 70,000 train, 2,500 validation, 2,500 test per category). Simplified using Ramer-Douglas-Peucker at $\epsilon = 4.0$ with maximum sequence length of 3,072 bits.
- **Tier C (Tabler Icons):** 5,130 vector icons designed on a 24x24 grid. Used as an analytical oracle to compute compression ceilings for geometric repetition and subroutine calls.
- **Tier D (FS-COCO):** 10,000 complex scene sketches averaging 62 strokes and 2,440 points. Used to analyze multi-scale compositional hierarchy.
